"""The argus TUI: a keyboard-driven client of the headless core.

Nothing here runs the agent differently from ``argus run``: the TUI builds an
:class:`~argus.agent.Agent`, runs it in a worker thread, and shows what its
:class:`~argus.agent.Reporter` callbacks report. Callbacks go into one queue that
the UI drains many times a second, so streamed tokens and tool calls stay in
order and the UI never blocks the model. Approvals are the one blocking call:
the worker waits on a modal until you answer.

Keys: enter send · tab plan/build · ctrl+o model · ctrl+r show/hide thinking ·
ctrl+t theme · ctrl+s sessions · ctrl+n new session · ctrl+d diff of the last run ·
esc interrupt · ctrl+q quit. Slash commands: type / in the prompt (/help lists them).
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
from textual.suggester import SuggestFromList
from textual.widgets import Button, Footer, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from argus.agent import Agent, Reporter, RunResult, brief_args, first_line
from argus.approval import Approver, Request
from argus.config import Config
from argus.tools.jobs import Jobs
from argus.tui import themes
from argus.tui.live import Meter, constrained_view

CHANGED = re.compile(r"\[files changed: (.*?)\]")

COMMANDS = {
    "/model": "switch model: /model opus, /model ollama/qwen3-coder:30b (no argument: pick)",
    "/models": "pick a model from the list",
    "/new": "start a new session",
    "/sessions": "open an earlier session",
    "/plan": "plan mode: read-only exploration ending in a plan",
    "/build": "build mode: the agent may edit and run commands",
    "/think": "show or hide the model's thinking",
    "/jobs": "background jobs and their latest output",
    "/diff": "what the last run changed",
    "/theme": "next theme, or /theme NAME",
    "/clear": "clear the transcript (the session keeps its history)",
    "/help": "this list",
    "/quit": "quit",
}


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
        # tool-call arguments count in the meter but are not shown as text
        text = d.text if d.kind != "tool" else ""
        self._put("delta", d.kind, text, d.completion.completion_tokens)

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
    """A filterable list. With a placeholder it also takes free text (model specs):
    enter picks the highlighted entry, or uses what was typed when it looks like a spec
    (has a / or @) or nothing matches."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "close"),
        Binding("down", "move(1)", "down", show=False),
        Binding("up", "move(-1)", "up", show=False),
    ]

    def __init__(
        self,
        title: str,
        options: list[tuple[str, Any]],
        placeholder: str = "",
        subtitle: str = "",
    ):
        super().__init__()
        self.title_text = title
        self.subtitle = subtitle
        # (value, label, plain text for filtering); a label may be rich Text
        self.options = [(v, lbl, str(getattr(lbl, "plain", lbl)).lower()) for v, lbl in options]
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label(self.title_text, id="picker-title")
            if self.subtitle:
                yield Label(self.subtitle, classes="dim")
            if self.placeholder:
                yield Input(placeholder=self.placeholder, id="picker-input")
            yield OptionList(*[Option(lbl, id=v) for v, lbl, _ in self.options], id="picker-list")

    def on_mount(self) -> None:
        if self.options:
            self.query_one(OptionList).highlighted = 0

    def action_move(self, step: int) -> None:
        lst = self.query_one(OptionList)
        if lst.option_count:
            lst.highlighted = ((lst.highlighted or 0) + step) % lst.option_count

    def on_input_changed(self, event: Input.Changed) -> None:
        needle = event.value.lower().strip()
        lst = self.query_one(OptionList)
        lst.clear_options()
        lst.add_options([Option(lbl, id=v) for v, lbl, plain in self.options if needle in plain])
        if lst.option_count:
            lst.highlighted = 0

    def on_input_submitted(self, event: Input.Submitted) -> None:
        typed = event.value.strip()
        lst = self.query_one(OptionList)
        picked = None
        if lst.option_count and lst.highlighted is not None:
            picked = lst.get_option_at_index(lst.highlighted).id
        if typed and ("/" in typed or "@" in typed or picked is None):
            self.dismiss(typed)
        else:
            self.dismiss(picked or typed or None)

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
        Binding("ctrl+r", "toggle_thinking", "thinking"),
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
        self.jobs = Jobs()  # background jobs outlive model and mode switches
        self.agent: Agent | None = None
        self.running = False
        self.runs_done = 0
        self.last_run: str | None = None
        self.show_thinking = self.cfg.tui.show_reasoning
        self.stream: dict[str, Static] = {}  # current turn's streaming widgets
        self.stream_text: dict[str, str] = {}
        self.tools: list[tuple[Any, str]] = []  # (widget, header) of calls in progress
        self.todo_items: list[dict[str, Any]] = []
        self.changed: list[str] = []
        self.meter = Meter()
        self.totals = {"prompt": 0, "gen": 0, "cost": 0.0, "ctx": 0}

    @property
    def main(self) -> Any:
        """The main screen: widgets are looked up there even while a modal is open."""
        return self.screen_stack[0] if self.screen_stack else self.screen

    # -- layout -----------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Horizontal(id="body"):
            yield VerticalScroll(id="transcript")
            with Vertical(id="sidebar"):
                yield Static(id="meter")
                yield Static("todo", classes="side-title")
                yield Static("—", id="todo")
                yield Static("changed", classes="side-title")
                yield Static("—", id="changed")
                yield Static("jobs", classes="side-title")
                yield Static("—", id="jobs")
                yield Static("session", classes="side-title")
                yield Static("new", id="session")
        yield Input(
            placeholder="Ask argus…  (enter send · / commands · ctrl+o model · tab plan/build)",
            id="prompt",
            suggester=SuggestFromList(self._suggestions(), case_sensitive=False),
        )
        yield Footer()

    def _suggestions(self) -> list[str]:
        from argus.profiles import load_aliases

        try:
            aliases = list(load_aliases())
        except Exception:  # a broken user models.toml must not stop the TUI
            aliases = []
        return list(COMMANDS) + [f"/model {a}" for a in aliases]

    def on_mount(self) -> None:
        for t in themes.EXTRA:
            self.register_theme(t)
        self.theme = themes.initial(self.cfg.tui.theme, set(self.available_themes))
        self.theme_changed_signal.subscribe(self, lambda _: self.refresh_status())
        self.main.query_one("#meter", Static).border_title = "tokens"
        self.main.query_one("#transcript").set_class(not self.show_thinking, "hide-thinking")
        self.set_interval(1 / 20, self.drain)
        self.set_interval(0.2, self.refresh_meter)
        self.set_interval(2.0, self.refresh_jobs)
        self.agent = self.build_agent()
        if self.session_id:
            self.load_session(self.session_id)
        self.refresh_status()
        self.refresh_meter()
        self.main.query_one("#prompt", Input).focus()

    def build_agent(self) -> Agent:
        cfg = copy.deepcopy(self.cfg)
        cfg.agent.mode = self.mode
        agent = Agent(
            cfg, reporter=TUIReporter(self.events), approver=self.approver, jobs=self.jobs
        )
        for note in agent.notes:
            self.add_line(note, "note")
        self.meter.window = agent.context_window() or 0
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
        self.main.query_one("#status", Static).update(text)
        sid = self.session_id or "new"
        self.main.query_one("#session", Static).update(Text(f"{sid}\n{self.theme}"))

    def refresh_meter(self) -> None:
        theme = self.current_theme
        styles = {
            "phase": f"bold {theme.accent or theme.primary}",
            "phase-idle": "dim",
            "big": "bold",
            "dim": "dim",
            "warn": f"bold {theme.warning}",
            "spark": str(theme.primary),
        }
        text = Text()
        for i, (line, style) in enumerate(self.meter.lines()):
            text.append(("\n" if i else "") + line, style=styles.get(style, ""))
        self.main.query_one("#meter", Static).update(text)

    def refresh_jobs(self) -> None:
        if not self.jobs.jobs:
            return
        remote = self.agent is not None and self.agent.executor.kind != "local"
        if remote and not self.running:
            return  # over SSH every poll is a round trip; tool events refresh the list
        lines = [
            f"{j.id} {'●' if j.status() == 'running' else '○'} {first_line(j.cmd, 26)}"
            for j in self._jobs()
        ]
        self.main.query_one("#jobs", Static).update(Text("\n".join(lines[-8:]) or "—"))

    def _jobs(self) -> list[Any]:
        for j in self.jobs.jobs.values():
            self.jobs.refresh(j)
        return list(self.jobs.jobs.values())

    # -- transcript -------------------------------------------------------------------------------------

    def add_line(self, text: str | Text, cls: str, markdown: bool = False) -> Static | Markdown:
        box = self.main.query_one("#transcript", VerticalScroll)
        widget: Static | Markdown
        if markdown:
            widget = Markdown(str(text), classes=cls)
        else:
            widget = Static(text if isinstance(text, Text) else Text(text), classes=cls)
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
        self.meter.start_turn(turn, agent)

    @property
    def constrained(self) -> bool:
        return self.agent is not None and self.agent.protocol.name != "native"

    def _show(self, agent: str, kind: str, text: str) -> None:
        """Update (or start) this turn's streaming widget for thinking or text."""
        key = f"{agent}:{kind}:view"
        if key not in self.stream:
            self.stream[key] = self.add_line(
                "", "reasoning" if kind == "reasoning" else "assistant"
            )  # type: ignore[assignment]
        if kind == "reasoning":
            body = Text()
            body.append(self._indent(agent, "∴ thinking\n"), style="bold")
            body.append(text)
            self.stream[key].update(body)
        else:
            self.stream[key].update(Text(self._indent(agent, text)))
        self.main.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def on_agent_delta(self, agent: str, kind: str, text: str, tokens: int = 0) -> None:
        self.meter.stream(kind, tokens, agent)
        if not text:
            return
        key = f"{agent}:{kind}"
        self.stream_text[key] = self.stream_text.get(key, "") + text
        raw = self.stream_text[key]
        if kind == "content" and self.constrained:
            # a JSON action: show its thought (and a <think> prefix) as they stream
            think, saying = constrained_view(raw)
            if think:
                self._show(agent, "reasoning", think)
            if saying:
                self._show(agent, "content", saying)
            return
        self._show(agent, kind, raw)

    def on_agent_turn_end(
        self, agent: str, turn: int, prompt: int, gen: int, final: str | None, visible: str
    ) -> None:
        self.meter.end_turn(prompt, gen, agent)
        if not agent:
            self.totals["prompt"] += prompt
            self.totals["gen"] += gen
            self.totals["ctx"] = prompt + gen
        streamed = self.stream.get(f"{agent}:content:view")
        shown = self.stream_text.get(f"{agent}:content", "")
        if streamed is not None and (final is not None or self.constrained or shown != visible):
            # a final answer is shown rendered; a JSON action (constrained protocols) is
            # replaced by its thought, the call itself shows as a tool line
            streamed.remove()
            if final is None and visible.strip():
                self.add_line(self._indent(agent, visible.strip()), "assistant")
        if final is not None and not agent:
            self.add_line(final, "final", markdown=True)
        self.stream, self.stream_text = {}, {}
        self.refresh_status()

    def on_agent_tool_start(self, agent: str, name: str, args: str) -> None:
        self.meter.tool(name, agent)
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
        self.meter.phase = "waiting"
        files = [self.relative(path)] if path else []
        m = CHANGED.search(text)
        if m:
            files += [f.split(" ", 1)[-1] for f in m.group(1).split(", ")]
        for f in files:
            if f and f not in self.changed:
                self.changed.append(f)
        if files:
            self.main.query_one("#changed", Static).update(Text("\n".join(self.changed[-15:])))
        if name in ("bash", "job"):
            self.refresh_jobs()

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
        self.main.query_one("#todo", Static).update(Text(text))

    def on_agent_run_end(self, agent: str, result: RunResult) -> None:
        if agent:
            self.add_line(
                self._indent(agent, f"done: {result.status}, {result.turns} turns"), "note"
            )
            return
        if result.cost_usd:
            self.totals["cost"] += result.cost_usd
            self.meter.cost = self.totals["cost"]
        if result.status not in ("completed", "refused"):  # a refusal was shown as it came
            self.add_line(
                f"run {result.status}" + (f": {result.error}" if result.error else ""), "failure"
            )

    def on_agent_done(self, agent: str, result: RunResult | None, error: str | None) -> None:
        self.running = False
        self.runs_done += 1
        self.meter.stop()
        if error:
            self.add_line(f"error: {error}", "failure")
        if result is not None and self.agent is not None:
            row = self.agent.store.run(result.run_id)
            self.session_id = row["session_id"] or self.session_id
        self.refresh_status()
        self.refresh_meter()
        self.refresh_jobs()

    # -- sending ----------------------------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        if text.startswith("/"):
            event.input.value = ""
            self.command(text)
            return
        if not text or self.running:
            if self.running:
                self.notify("still working; esc interrupts", severity="warning")
            return
        event.input.value = ""
        self.add_line(f"› {text}", "user")
        self.running = True
        self.meter.start_run()
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

    def command(self, text: str) -> None:
        """Slash commands typed in the prompt."""
        name, _, arg = text.partition(" ")
        name, arg = name.lower(), arg.strip()
        if name == "/model" and arg:
            self.switch_model(arg)
        elif name in ("/model", "/models"):
            self.action_pick_model()
        elif name == "/new":
            self.action_new_session()
        elif name == "/sessions":
            self.action_sessions()
        elif name in ("/plan", "/build"):
            if (name == "/plan") != (self.mode == "plan"):
                self.action_toggle_mode()
        elif name == "/think":
            self.action_toggle_thinking()
        elif name == "/jobs":
            self.show_jobs()
        elif name == "/diff":
            self.action_diff()
        elif name == "/theme":
            if arg and arg in self.available_themes:
                self.theme = arg
                themes.save_theme(arg)
            elif arg:
                self.notify(f"no theme {arg!r}", severity="error")
            else:
                self.action_next_theme()
        elif name == "/clear":
            self.main.query_one("#transcript", VerticalScroll).remove_children()
        elif name == "/quit":
            self.run_action("quit")
        elif name == "/help":
            body = Text()
            for cmd, what in COMMANDS.items():
                body.append(f"{cmd:<10}", style="bold")
                body.append(f" {what}\n")
            body.append("\nkeys: ", style="bold")
            body.append(
                "enter send · tab plan/build · ctrl+o model · ctrl+r thinking · ctrl+t theme · "
                "ctrl+s sessions · ctrl+n new · ctrl+d diff · esc interrupt · ctrl+p palette · "
                "ctrl+q quit"
            )
            self.push_screen(TextScreen("commands (esc to close)", body))
        else:
            self.notify(f"unknown command {name}; /help lists them", severity="warning")

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
        self.jobs.close()

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

    def action_toggle_thinking(self) -> None:
        self.show_thinking = not self.show_thinking
        self.main.query_one("#transcript").set_class(not self.show_thinking, "hide-thinking")
        self.notify("thinking shown" if self.show_thinking else "thinking hidden")

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
            if self.running:
                self.notify("switch models between runs (esc interrupts)", severity="warning")
            return
        from argus.choices import model_choices

        store = self.agent.store if self.agent is not None else None
        choices = model_choices(store)
        w_name = min(max((len(c.name) for c in choices), default=8), 22)
        w_spec = min(max((len(c.spec) for c in choices), default=20), 56)
        options = []
        for c in choices:
            label = Text(no_wrap=True, overflow="ellipsis")
            label.append("● " if c.ready else "○ ", style="green" if c.ready else "dim")
            label.append(f"{c.name[:w_name]:<{w_name}}", style="bold" if c.ready else "dim")
            label.append(f"  {c.spec[:w_spec]:<{w_spec}}", style="dim")
            label.append(f"  {c.group:<6}", style="italic")
            if c.note:
                label.append(f"  {c.note}", style="dim" if c.ready else "yellow")
            options.append((c.spec, label))
        where = self.agent.llm.describe() if self.agent is not None else ""
        self.push_screen(
            PickerScreen(
                "switch model",
                options,
                "filter, or type a spec: anthropic/claude-sonnet-5, ollama/qwen3@http://host:11434/v1",
                subtitle=f"now: {self.cfg.model.model} · {where}   ● ready  ○ needs a key or "
                "server   the session carries over",
            ),
            self.switch_model,
        )

    def switch_model(self, spec: str | None) -> None:
        if not spec:
            return
        if self.running:
            self.notify("switch models between runs (esc interrupts)", severity="warning")
            return
        try:
            cfg = self.make_config(spec)
        except Exception as e:
            self.notify(f"cannot use {spec}: {e}", severity="error")
            return
        old_cfg = self.cfg
        self.cfg = cfg
        try:
            self.agent = self.rebuild(self.agent)
        except Exception as e:
            self.cfg = old_cfg
            self.agent = self.build_agent()
            self.notify(f"cannot use {spec}: {e}", severity="error")
            return
        self.add_line(
            f"· model: {cfg.model.model} ({self.agent.llm.describe()}); the session continues",
            "note",
        )
        self.refresh_status()

    def show_jobs(self) -> None:
        jobs = self._jobs()
        if not jobs:
            self.notify("no background jobs")
            return
        body = Text()
        for j in jobs:
            body.append(f"job {j.id} · {j.status()} · pid {j.handle.pid}\n", style="bold")
            body.append(f"$ {j.cmd}\n", style="dim")
            try:
                tail = j.handle.read(0).decode("utf-8", "replace").splitlines()[-12:]
            except Exception as e:  # e.g. the SSH connection is gone
                tail = [f"(output unavailable: {e})"]
            body.append("\n".join(tail) + "\n\n")
        self.push_screen(TextScreen("background jobs (esc to close)", body))

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
        box = self.main.query_one("#transcript", VerticalScroll)
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
        self.main.query_one("#transcript", VerticalScroll).remove_children()
        self.totals = {"prompt": 0, "gen": 0, "cost": 0.0, "ctx": 0}
        self.meter = Meter(window=self.meter.window)
        self.changed, self.todo_items = [], []
        self.main.query_one("#todo", Static).update("—")
        self.main.query_one("#changed", Static).update("—")
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
        self.jobs.close()
        self.exit()
