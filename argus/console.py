"""Terminal output: live progress on stderr, transcripts for ``argus show``."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

from argus.agent import Reporter, RunResult, brief_args
from argus.protocols import Parsed, ToolCall
from argus.providers.base import Completion, Delta
from argus.tools import ToolResult


class Style:
    def __init__(self, stream: TextIO, color: bool | None = None):
        if color is None:
            color = stream.isatty() and not os.environ.get("NO_COLOR")
        self.color = color

    def _w(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.color else s

    def dim(self, s: str) -> str:
        return self._w("2", s)

    def bold(self, s: str) -> str:
        return self._w("1", s)

    def red(self, s: str) -> str:
        return self._w("31", s)

    def green(self, s: str) -> str:
        return self._w("32", s)

    def yellow(self, s: str) -> str:
        return self._w("33", s)

    def cyan(self, s: str) -> str:
        return self._w("36", s)


def first_line(text: str, limit: int = 100) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


class ConsoleReporter(Reporter):
    def __init__(
        self,
        stream: TextIO | None = None,
        verbose: bool = False,
        color: bool | None = None,
        prefix: str = "",
    ):
        self.out = stream or sys.stderr
        self.verbose = verbose
        self.s = Style(self.out, color)
        self._streaming: str | None = None
        self.prefix = prefix  # subagent output is indented under the parent's

    def _p(self, text: str) -> None:
        self._end_stream()
        if self.prefix:
            text = "\n".join(self.prefix + line for line in text.split("\n"))
        print(text, file=self.out, flush=True)

    def child(self, name: str) -> ConsoleReporter:
        return ConsoleReporter(
            self.out, self.verbose, self.s.color, prefix=self.prefix + self.s.dim(f"{name} │ ")
        )

    def todos(self, turn: int, items: list[dict[str, Any]]) -> None:
        from argus.tools.todo import MARKS

        for i in items:
            line = f"    {MARKS[i['status']]} {i['content']}"
            self._p(self.s.dim(line) if i["status"] == "done" else line)

    def hook(self, event: str, result: Any) -> None:
        if result.blocked or result.error or not result.ok:
            why = result.error or first_line(result.stderr) or f"exit {result.code}"
            self._p(self.s.yellow(f"    ↯ {event} hook: {why}"))

    def _end_stream(self) -> None:
        if self._streaming:
            print(file=self.out, flush=True)
            self._streaming = None

    def run_start(self, run_id: str, task: str) -> None:
        self._p(self.s.dim(f"run {run_id}: {first_line(task, 120)}"))

    def delta(self, d: Delta) -> None:
        if not self.verbose or d.kind == "tool":
            return
        if self._streaming != d.kind:
            self._end_stream()
            self._streaming = d.kind
        text = self.s.dim(d.text) if d.kind == "reasoning" else d.text
        self.out.write(text)
        self.out.flush()

    def turn_end(self, turn: int, c: Completion, p: Parsed) -> None:
        stats = f"{c.prompt_tokens}→{c.completion_tokens} tok, {c.total_ms / 1000:.1f}s"
        if c.cached_tokens:
            stats += f", cache {c.cached_tokens}"
        if c.aborted:
            stats += f", aborted: {c.aborted}"
        elif c.finish_reason == "length":
            stats += ", hit max_tokens"
        self._p(self.s.dim(f"[{turn}] {stats}"))
        if not self.verbose and c.content.strip() and p.calls:
            self._p(self.s.dim("    " + first_line(c.content)))

    def tool_start(self, turn: int, call: ToolCall) -> None:
        self._p(f"  {self.s.cyan(call.name or '?')}({brief_args(call.args)})")

    def tool_end(self, turn: int, call: ToolCall, result: ToolResult, ms: float) -> None:
        mark = self.s.green("✓") if result.ok else self.s.red("✗")
        self._p(f"    {mark} {first_line(result.text)} {self.s.dim(f'({ms:.0f}ms)')}")

    def failure(self, turn: int | None, tag: str, detail: str) -> None:
        if tag == "refusal":
            self._p(self.s.yellow(f"  ⊘ the model declined: {first_line(detail, 160)}"))
            self._p(
                self.s.dim(
                    "    the session continues: argus run --continue 'rephrased task' "
                    "(add -m SPEC for another model)"
                )
            )
            return
        self._p(self.s.yellow(f"  ! {tag}: {first_line(detail, 160)}"))

    def note(self, text: str) -> None:
        self._p(self.s.dim(f"  · {text}"))

    def approval(self, turn: int, call: ToolCall, decision: Any) -> None:
        if not decision.allow:
            self._p(self.s.yellow(f"    ⊘ {call.name} denied: {decision.reason}"))
        elif decision.asked:
            self._p(self.s.dim(f"    ✓ approved ({decision.answer})"))

    def run_end(self, r: RunResult) -> None:
        color = {"completed": self.s.green, "refused": self.s.yellow}.get(r.status, self.s.red)
        tags = f" [{', '.join(r.tags)}]" if r.failures else ""
        cost = f" · ${r.cost_usd:.4f}" if r.cost_usd is not None else ""
        self._p(
            f"{color(r.status)}{tags} · {r.turns} turns · {r.tool_calls} calls · "
            f"{r.completion_tokens} gen tok · max ctx {r.max_context} · {r.wall_ms / 1000:.1f}s{cost} · "
            f"run {r.run_id}"
        )
        if r.error:
            self._p(self.s.red(r.error))


def _clip(text: str | None, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more chars]"


def render_run(store: Any, run_id: str, full: bool = False, color: bool | None = None) -> str:
    s = Style(sys.stdout, color)
    run = store.run(run_id)
    limit = 0 if full else 1500
    lines = [
        s.bold(f"run {run['id']}") + f"  {run['status']}",
        f"task:      {first_line(run['task'], 200)}",
        f"config:    {run['config_name']} ({run['config_hash']})  protocol={run['protocol']}  model={run['model']}",
        f"provider:  {run['provider'] or '-'}",
        f"workspace: {run['executor']}",
        f"tokens:    prompt Σ{run['prompt_tokens']}  gen Σ{run['completion_tokens']}  "
        f"reasoning Σ{run['reasoning_tokens']}  max ctx {run['max_context_tokens']}"
        + (f"  overhead {run['overhead_tokens']}" if run["overhead_tokens"] is not None else "")
        + (f"  cache read Σ{run['cache_read_tokens']}" if run["cache_read_tokens"] else "")
        + (f"  cache write Σ{run['cache_write_tokens']}" if run["cache_write_tokens"] else "")
        + (f"  cost ${run['cost_usd']:.4f}" if run["cost_usd"] is not None else ""),
        f"time:      wall {(run['wall_ms'] or 0) / 1000:.1f}s  llm {(run['llm_ms'] or 0) / 1000:.1f}s  "
        f"tools {(run['tool_ms'] or 0) / 1000:.1f}s",
    ]
    if run["batch_id"]:
        lines.append(f"batch:     {run['batch_id']} variant={run['variant']} task={run['task_id']}")
    if run["check_passed"] is not None:
        lines.append(f"check:     {'passed' if run['check_passed'] else 'FAILED'}")
    if run["parent_run_id"]:
        lines.append(
            f"subagent:  {run['agent']} of run {run['parent_run_id']} (turn {run['parent_turn']})"
        )
    if run["mode"] == "plan":
        lines.append("mode:      plan (read-only)")
    children = store.children(run_id)
    for ch in children:
        lines.append(
            f"subagent:  {ch['agent']} → run {ch['id']} ({ch['status']}, {ch['n_turns']} turns)"
        )
    todos = store.events(run_id, "todo")
    if todos:
        from argus.tools.todo import render

        lines.append(
            "todo:      "
            + render(json.loads(todos[-1]["data_json"])).replace("\n", "\n           ")
        )
    fails = store.failures(run_id)
    if fails:
        lines.append("failures:  " + ", ".join(f"{f['tag']}@{f['turn_idx']}" for f in fails))
    if run["error"]:
        lines.append(s.red(f"error:     {run['error']}"))
    calls_by_turn: dict[int, list[Any]] = {}
    for tc in store.tool_calls(run_id):
        calls_by_turn.setdefault(tc["turn_idx"], []).append(tc)
    fails_by_turn: dict[int, list[Any]] = {}
    for f in fails:
        fails_by_turn.setdefault(f["turn_idx"], []).append(f)
    for t in store.turns(run_id):
        head = (
            f"── turn {t['idx']}"
            + (f".{t['attempt']}" if t["attempt"] else "")
            + f" · prompt {t['prompt_tokens']} (cache {t['cached_tokens'] or 0}) · gen {t['completion_tokens']}"
            + f" · reasoning≈{t['reasoning_tokens']} · {(t['total_ms'] or 0) / 1000:.2f}s"
            + (f" · ttft {t['ttft_ms']:.0f}ms" if t["ttft_ms"] else "")
            + (f" · {t['gen_tps']:.1f} tok/s" if t["gen_tps"] else "")
            + f" · finish={t['finish_reason']}"
            + (f" · ABORTED {t['aborted']}" if t["aborted"] else "")
        )
        lines.append("")
        lines.append(s.bold(head))
        if t["reasoning"]:
            lines.append(s.dim(_clip(t["reasoning"], limit)))
        if t["content"]:
            lines.append(_clip(t["content"], limit))
        if not t["attempt"]:
            for tc in calls_by_turn.get(t["idx"], []):
                args = json.loads(tc["args_json"]) if tc["args_json"] else {}
                mark = s.green("✓") if tc["ok"] else s.red("✗")
                lines.append(
                    f"{mark} {s.cyan(tc['name'])}({brief_args(args, 200)})  {(tc['duration_ms'] or 0):.0f}ms"
                )
                lines.append("  " + _clip(tc["result"], limit).replace("\n", "\n  "))
            for f in fails_by_turn.get(t["idx"], []):
                lines.append(s.yellow(f"! {f['tag']}: {f['detail']}"))
    if run["final_answer"]:
        lines += ["", s.bold("final answer:"), run["final_answer"]]
    return "\n".join(lines)
