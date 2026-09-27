"""The argus TUI: a keyboard-driven client of the headless core.

Nothing here runs the agent differently from ``argus run``: the TUI builds an
:class:`~argus.agent.Agent`, runs it in a worker thread, and shows what its
:class:`~argus.agent.Reporter` callbacks report. Callbacks go into one queue that
the UI drains a few times per second, so streamed tokens and tool calls stay in
order and the UI never blocks the model. Approvals are the one blocking call:
the worker waits on a modal until you answer.

Keys: enter send · tab plan/build · ctrl+o model · ctrl+t theme · ctrl+s sessions ·
ctrl+n new session · ctrl+d diff of the last run · esc interrupt · ctrl+q quit.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from argus.agent import Agent, Reporter, RunResult, brief_args, first_line
from argus.approval import Approver, Request
from argus.config import Config
from argus.tui import themes

CHANGED = re.compile(r"\[files changed: (.*?)\]")


class TUIReporter(Reporter):
    """Queues every callback for the UI thread (the agent runs in a worker thread)."""

    def __init__(self, events: deque, agent: str = ""):
        self.events = events
        self.agent = agent  # subagent name, "" for the main agent

    def _put(self, kind: str, *payload: Any) -> None:
        self.events.append((kind, self.agent, payload))

    def run_start(self, run_id: str, task: str) -> None:
        self._put("run_start", run_id, task)

    def turn_start(self, turn: int) -> None:
        self._put("turn_start", turn)

    def delta(self, d: Any) -> None:
        if d.kind != "tool":
            self._put("delta", d.kind, d.text)

    def turn_end(self, turn: int, c: Any, p: Any) -> None:
        # p.content is the visible text (a JSON action's "thought" under constrained protocols)
        self._put("turn_end", turn, c.prompt_tokens, c.completion_tokens, p.final, p.content)

    def tool_start(self, turn: int, call: Any) -> None:
        self._put("tool_start", call.name, brief_args(call.args))

    def tool_end(self, turn: int, call: Any, result: Any, ms: float) -> None:
        path = result.meta.get("path") if call.name in ("edit", "write") and result.ok else None
        self._put("tool_end", call.name, result.ok, result.text, ms, path)

    def failure(self, turn: int | None, tag: str, detail: str) -> None:
        self._put("failure", tag, detail)

    def note(self, text: str) -> None:
        self._put("note", text)

    def todos(self, turn: int, items: list[dict[str, Any]]) -> None:
        self._put("todos", items)

    def approval(self, turn: int, call: Any, decision: Any) -> None:
        if not decision.allow:
            self._put("note", f"{call.name} denied: {decision.reason}")

    def run_end(self, result: RunResult) -> None:
        self._put("run_end", result)

    def child(self, name: str) -> TUIReporter:
        return TUIReporter(self.events, name)


class TUIApprover(Approver):
    """Asks with a modal; the agent's worker thread waits for the answer."""

    interactive = True

    def __init__(self, app: ArgusApp):
        self.app = app

    def ask(self, req: Request) -> str:
        done = threading.Event()
        answer: dict[str, str] = {}

        def answered(value: str | None) -> None:
            answer["a"] = value or "no"
            done.set()

        self.app.pending_answers.append(answered)
        try:
            self.app.call_from_thread(self.app.ask_approval, req, answered)
        except RuntimeError:  # the app is gone
            return "no"
        done.wait()
        return answer.get("a", "no")


# -- screens ----------------------------------------------------------------------------------------


class ApprovalScreen(ModalScreen[str]):
    BINDINGS = [
        Binding("y", "answer('yes')", "yes"),
        Binding("a", "answer('always')", "always"),
        Binding("n", "answer('no')", "no"),
        Binding("escape", "answer('no')", "no"),
    ]

    def __init__(self, req: Request):
        super().__init__()
        self.req = req

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Label("argus wants to run", classes="dim")
            yield Static(Text(self.req.summary), id="approval-summary")
            yield Label(Text(self.req.reason), classes="dim")
            with Horizontal(id="approval-buttons"):
                yield Button("[y] yes", id="yes", variant="success")
                yield Button("[a] always", id="always", variant="primary")
                yield Button("[n] no", id="no", variant="error")

    def action_answer(self, value: str) -> None:
        self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "no")


class PickerScreen(ModalScreen[str | None]):
    """A filterable list with an input for free text (model specs)."""

    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def __init__(self, title: str, options: list[tuple[str, str]], placeholder: str = ""):
        super().__init__()
        self.title_text = title
        self.options = options  # (value, label)
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label(self.title_text, id="picker-title")
            if self.placeholder:
                yield Input(placeholder=self.placeholder, id="picker-input")
            yield OptionList(
                *[Option(label, id=value) for value, label in self.options], id="picker-list"
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        needle = event.value.lower()
        lst = self.query_one(OptionList)
        lst.clear_options()
        lst.add_options(
            [Option(label, id=v) for v, label in self.options if needle in label.lower()]
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class TextScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "close"), Binding("q", "dismiss(None)", "close")]

    def __init__(self, title: str, body: Any):
        super().__init__()
        self.title_text = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="text-screen"):
            yield Label(self.title_text, id="picker-title")
            with VerticalScroll():
                yield Static(self.body)


# -- the app ------------------------------------------------------------------------------------------


class ArgusApp(App):
    CSS_PATH = "argus.tcss"
    TITLE = "argus"
    BINDINGS = [
        Binding("ctrl+q", "quit", "quit", priority=True),
        Binding("tab", "toggle_mode", "plan/build", priority=True),
        Binding("ctrl+o", "pick_model", "model"),
        Binding("ctrl+t", "next_theme", "theme"),
        Binding("ctrl+s", "sessions", "sessions"),
        Binding("ctrl+n", "new_session", "new"),
        Binding("ctrl+d", "diff", "diff"),
        Binding("escape", "interrupt", "interrupt"),
    ]

    def __init__(
        self,
        make_config: Callable[[str | None], Config],
        *,
        session: str | None = None,
        approver: Approver | None = None,
    ):
        super().__init__()
        self.make_config = make_config  # model spec (None = as configured) -> Config
        self.cfg = make_config(None)
        self.mode = self.cfg.agent.mode
        self.session_id = session
        self.events: deque = deque()
        self.pending_answers: list[Callable[[str | None], None]] = []
        self.approver = approver or TUIApprover(self)
        self.agent: Agent | None = None
        self.running = False
        self.runs_done = 0
        self.last_run: str | None = None
        self.stream: dict[str, Static] = {}  # current turn's streaming widgets
        self.stream_text: dict[str, str] = {}
        self.tools: list[tuple[Any, str]] = []  # (widget, header) of calls in progress
        self.todo_items: list[dict[str, Any]] = []
        self.changed: list[str] = []
        self.totals = {"prompt": 0, "gen": 0, "cost": 0.0, "ctx": 0}

    # -- layout -----------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Horizontal(id="body"):
            yield VerticalScroll(id="transcript")
            with Vertical(id="sidebar"):
                yield Static("todo", classes="side-title")
                yield Static("—", id="todo")
                yield Static("changed", classes="side-title")
                yield Static("—", id="changed")
                yield Static("session", classes="side-title")
                yield Static("new", id="session")
        yield Input(
            placeholder="Ask argus…  (enter to send · tab plan/build · ctrl+o model)", id="prompt"
        )
        yield Footer()

    def on_mount(self) -> None:
        for t in themes.EXTRA:
            self.register_theme(t)
        self.theme = themes.initial(self.cfg.tui.theme, set(self.available_themes))
        self.theme_changed_signal.subscribe(self, lambda _: self.refresh_status())
        self.set_interval(1 / 20, self.drain)
        self.agent = self.build_agent()
        if self.session_id:
            self.load_session(self.session_id)
        self.refresh_status()
        self.query_one("#prompt", Input).focus()

    def build_agent(self) -> Agent:
        cfg = copy.deepcopy(self.cfg)
        cfg.agent.mode = self.mode
        agent = Agent(cfg, reporter=TUIReporter(self.events), approver=self.approver)
        for note in agent.notes:
            self.add_line(note, "note")
        return agent

    def refresh_status(self) -> None:
        a = self.agent
        mode = "PLAN" if self.mode == "plan" else "BUILD"
        policy = a.cfg.agent.approval if a else self.cfg.agent.approval
        provider = a.llm.describe() if a else self.cfg.model.provider
        t = self.totals
        ctx = ""
        n_ctx = a.context_window() if a else 0
        if n_ctx and t["ctx"]:
            ctx = f" · ctx {100 * t['ctx'] / n_ctx:.0f}%"
        cost = f" · ${t['cost']:.4f}" if t["cost"] else ""
        state = " · working… (esc to stop)" if self.running else ""
        text = Text()
        text.append(" argus ", style="bold reverse")
        text.append(f" {self.cfg.model.model} ", style="bold")
        text.append(f"· {provider} ")
        text.append(f" {mode} ", style="bold reverse" if self.mode == "plan" else "bold")
        text.append(f" · {policy} · tokens {t['prompt']}→{t['gen']}{cost}{ctx}{state}")
        self.query_one("#status", Static).update(text)
        sid = self.session_id or "new"
        self.query_one("#session", Static).update(Text(f"{sid}\n{self.theme}"))

    # -- transcript -------------------------------------------------------------------------------------

    def add_line(self, text: str, cls: str, markdown: bool = False) -> Static | Markdown:
        box = self.query_one("#transcript", VerticalScroll)
        widget: Static | Markdown = (
            Markdown(text, classes=cls) if markdown else Static(Text(text), classes=cls)
        )
        box.mount(widget)
        box.scroll_end(animate=False)
        return widget

    def drain(self) -> None:
        while self.events:
            kind, agent, payload = self.events.popleft()
            getattr(self, f"on_agent_{kind}")(agent, *payload)

    def _indent(self, agent: str, text: str) -> str:
        return f"{agent} │ {text}" if agent else text

    def on_agent_run_start(self, agent: str, run_id: str, task: str) -> None:
        if agent:
            self.add_line(self._indent(agent, f"subagent: {first_line(task, 120)}"), "note")
        else:
            self.last_run = run_id

    def on_agent_turn_start(self, agent: str, turn: int) -> None:
        self.stream, self.stream_text = {}, {}

    def on_agent_delta(self, agent: str, kind: str, text: str) -> None:
        if kind == "reasoning" and not self.cfg.tui.show_reasoning:
            return
        key = f"{agent}:{kind}"
        if key not in self.stream:
            self.stream[key] = self.add_line(
                "", "reasoning" if kind == "reasoning" else "assistant"
            )  # type: ignore[assignment]
            self.stream_text[key] = ""
        self.stream_text[key] += text
        self.stream[key].update(Text(self._indent(agent, self.stream_text[key])))
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def on_agent_turn_end(
        self, agent: str, turn: int, prompt: int, gen: int, final: str | None, visible: str
    ) -> None:
        if not agent:
            self.totals["prompt"] += prompt
            self.totals["gen"] += gen
            self.totals["ctx"] = prompt + gen
        key = f"{agent}:content"
        streamed = self.stream.get(key)
        constrained = self.agent is not None and self.agent.protocol.name != "native"
        if streamed is not None and (
            final is not None or constrained or self.stream_text[key] != visible
        ):
            # a final answer is shown rendered; a JSON action (constrained protocols) is
            # replaced by its thought, the call itself shows as a tool line
            streamed.remove()
            if final is None and visible.strip() and visible != self.stream_text[key]:
                self.add_line(self._indent(agent, visible.strip()), "assistant")
        if final is not None and not agent:
            self.add_line(final, "final", markdown=True)
        self.stream, self.stream_text = {}, {}
        self.refresh_status()

    def on_agent_tool_start(self, agent: str, name: str, args: str) -> None:
        head = self._indent(agent, f"▸ {name}({args})")
        self.tools.append((self.add_line(head, "tool"), head))  # type: ignore[arg-type]

    def on_agent_tool_end(
        self, agent: str, name: str, ok: bool, text: str, ms: float, path: str | None
    ) -> None:
        widget, head = self.tools.pop() if self.tools else (self.add_line("", "tool"), name)
        mark = "✓" if ok else "✗"
        lines = text.strip().splitlines()
        preview = "\n".join(f"    {ln}" for ln in lines[:6])
        more = f"\n    … {len(lines) - 6} more lines" if len(lines) > 6 else ""
        widget.update(Text(f"{head} {mark} ({ms:.0f}ms)\n{preview}{more}"))
        widget.set_class(not ok, "tool-fail")
        files = [self.relative(path)] if path else []
        m = CHANGED.search(text)
        if m:
            files += [f.split(" ", 1)[-1] for f in m.group(1).split(", ")]
        for f in files:
            if f and f not in self.changed:
                self.changed.append(f)
        if files:
            self.query_one("#changed", Static).update(Text("\n".join(self.changed[-15:])))

    def relative(self, path: str) -> str:
        root = self.cfg.executor.workdir or os.getcwd()
        rel = os.path.relpath(path, root)
        return path if rel.startswith("..") else rel

    def on_agent_failure(self, agent: str, tag: str, detail: str) -> None:
        if tag == "refusal":
            self.add_line(
                self._indent(agent, f"⊘ the model declined: {first_line(detail, 200)}"), "refusal"
            )
            if not agent:
                self.add_line(
                    "the session continues: rephrase and send, or switch model with ctrl+o",
                    "note",
                )
            return
        self.add_line(self._indent(agent, f"! {tag}: {first_line(detail, 200)}"), "failure")

    def on_agent_note(self, agent: str, text: str) -> None:
        self.add_line(self._indent(agent, f"· {text}"), "note")

    def on_agent_todos(self, agent: str, items: list[dict[str, Any]]) -> None:
        from argus.tools.todo import MARKS

        self.todo_items = items
        text = "\n".join(f"{MARKS[i['status']]} {i['content']}" for i in items) or "—"
        self.query_one("#todo", Static).update(Text(text))

    def on_agent_run_end(self, agent: str, result: RunResult) -> None:
        if agent:
            self.add_line(
                self._indent(agent, f"done: {result.status}, {result.turns} turns"), "note"
            )
            return
        if result.cost_usd:
            self.totals["cost"] += result.cost_usd
        if result.status not in ("completed", "refused"):  # a refusal was shown as it came
            self.add_line(
                f"run {result.status}" + (f": {result.error}" if result.error else ""), "failure"
            )

    def on_agent_done(self, agent: str, result: RunResult | None, error: str | None) -> None:
        self.running = False
        self.runs_done += 1
        if error:
            self.add_line(f"error: {error}", "failure")
        if result is not None and self.agent is not None:
            row = self.agent.store.run(result.run_id)
            self.session_id = row["session_id"] or self.session_id
        self.refresh_status()

    # -- sending ----------------------------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        if not text or self.running:
            if self.running:
                self.notify("still working; esc interrupts", severity="warning")
            return
        event.input.value = ""
        self.add_line(f"› {text}", "user")
        self.running = True
        self.refresh_status()
        self.run_worker(lambda: self._run(text), thread=True, group="agent", name="run")

    def _run(self, text: str) -> None:
        result, error = None, None
        try:
            assert self.agent is not None
            result = self.agent.run(text, session=self.session_id)
        except Exception as e:  # shown in the transcript; the app keeps running
            error = f"{type(e).__name__}: {e}"
        self.events.append(("done", "", (result, error)))

    # -- approvals --------------------------------------------------------------------------------------

    def ask_approval(self, req: Request, answered: Callable[[str | None], None]) -> None:
        self.drain()  # show the call being asked about first

        def done(value: str | None) -> None:
            if answered in self.pending_answers:
                self.pending_answers.remove(answered)
            self.add_line(f"· {req.summary}: {value or 'no'}", "note")
            answered(value)

        self.push_screen(ApprovalScreen(req), done)

    def on_unmount(self) -> None:
        if self.agent is not None:
            self.agent.cancel()
        for answered in list(self.pending_answers):
            answered("no")

    # -- actions ----------------------------------------------------------------------------------------

    def _main_screen(self) -> bool:
        return len(self.screen_stack) == 1

    def action_toggle_mode(self) -> None:
        if not self._main_screen() or self.running:
            return
        self.mode = "build" if self.mode == "plan" else "plan"
        self.agent = self.rebuild(self.agent)
        self.add_line(
            "· plan mode: read-only exploration ending in a plan"
            if self.mode == "plan"
            else "· build mode: the agent may edit and run commands",
            "note",
        )
        self.refresh_status()

    def rebuild(self, old: Agent | None) -> Agent:
        if old is not None:
            old.close()
        return self.build_agent()

    def action_interrupt(self) -> None:
        if self.running and self.agent is not None:
            self.agent.cancel()
            self.add_line("· interrupting…", "note")

    def action_next_theme(self) -> None:
        names = [t for t in themes.ORDER if t in self.available_themes]
        i = names.index(self.theme) if self.theme in names else -1
        self.theme = names[(i + 1) % len(names)]
        themes.save_theme(self.theme)

    def action_pick_model(self) -> None:
        if not self._main_screen() or self.running:
            return
        from argus.profiles import load_profiles

        options = [
            (name, f"{name}  ·  {p.provider or 'local'}{'  ·  ' + p.notes if p.notes else ''}")
            for name, p in load_profiles().items()
        ]
        self.push_screen(
            PickerScreen(
                "model (enter a provider/model spec or pick a profile)", options, "provider/model"
            ),
            self.switch_model,
        )

    def switch_model(self, spec: str | None) -> None:
        if not spec:
            return
        try:
            cfg = self.make_config(spec)
        except Exception as e:
            self.notify(f"cannot use {spec}: {e}", severity="error")
            return
        self.cfg = cfg
        try:
            self.agent = self.rebuild(self.agent)
        except Exception as e:
            self.notify(f"cannot use {spec}: {e}", severity="error")
            return
        self.add_line(
            f"· model: {cfg.model.model} ({self.agent.llm.describe()}); the session continues",
            "note",
        )
        self.refresh_status()

    def action_sessions(self) -> None:
        if not self._main_screen() or self.running or self.agent is None:
            return
        rows = self.agent.store.sessions(limit=50)
        options = [
            (r["id"], f"{r['id']}  ·  {r['n_runs']} runs  ·  {r['title'] or ''}") for r in rows
        ]
        self.push_screen(PickerScreen("sessions", options), self.open_session)

    def open_session(self, session_id: str | None) -> None:
        if session_id:
            self.load_session(session_id)

    def load_session(self, session_id: str) -> None:
        assert self.agent is not None
        store = self.agent.store
        self.session_id = session_id
        box = self.query_one("#transcript", VerticalScroll)
        box.remove_children()
        for run in store.session_runs(session_id):
            for m in store.messages(run["id"]):
                if m["kind"] not in ("normal", None) or m["role"] == "system":
                    continue
                if m["role"] == "user":
                    self.add_line(f"› {m['content']}", "user")
                elif m["role"] == "assistant":
                    calls = json.loads(m["tool_calls_json"]) if m["tool_calls_json"] else []
                    if m["content"]:
                        self.add_line(
                            m["content"], "final" if not calls else "assistant", markdown=not calls
                        )
                    for c in calls:
                        fn = c.get("function") or {}
                        self.add_line(
                            f"▸ {fn.get('name')}({first_line(fn.get('arguments', ''), 80)})", "tool"
                        )
            self.last_run = run["id"]
        self.refresh_status()

    def action_new_session(self) -> None:
        if not self._main_screen() or self.running:
            return
        self.session_id = None
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.totals = {"prompt": 0, "gen": 0, "cost": 0.0, "ctx": 0}
        self.changed, self.todo_items = [], []
        self.query_one("#todo", Static).update("—")
        self.query_one("#changed", Static).update("—")
        self.refresh_status()

    def action_diff(self) -> None:
        if not self._main_screen() or not self.last_run or self.agent is None:
            return
        body: Any
        try:
            from argus.cli import _shadow

            rows = self.agent.store.checkpoints(self.last_run)
            if not rows:
                body = Text("no checkpoints for the last run")
            else:
                git = _shadow(self.agent.store, self.last_run)
                patch = git.patch(rows[0]["commit_sha"], git.snapshot())
                body = Syntax(patch or "(no changes)", "diff", theme="ansi_dark", word_wrap=True)
        except Exception as e:
            body = Text(f"cannot diff: {e}")
        self.push_screen(
            TextScreen(f"changes since run {self.last_run} started (esc to close)", body)
        )

    async def action_quit(self) -> None:
        if self.agent is not None:
            self.agent.cancel()
            self.agent.close()
        self.exit()
