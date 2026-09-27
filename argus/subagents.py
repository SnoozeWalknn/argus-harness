"""Subagents with isolated context (as in Claude Code): the ``task`` tool.

The parent hands a self-contained subtask to a subagent, which runs as a
separate argus run with a fresh context: its own system prompt, a tool subset,
optionally another model, and an approval policy no looser than the parent's.
Only the subagent's final answer (plus a one-line summary) comes back, so the
exploration it did never fills the parent's context. Subagent runs are ordinary
runs in the log, linked by ``parent_run_id`` / ``parent_turn``. Subagents cannot
start subagents.

Built-in agents: ``explore`` (read-only research) and ``general``. More come from
``*.md`` files in ``.argus/agents`` and ``.claude/agents`` in the workspace and
``~/.config/argus/agents``, with front matter::

    ---
    name: reviewer
    description: Reviews a diff for bugs and missing tests.
    tools: read, grep, glob, bash      # default: the parent's tools
    model: anthropic/claude-haiku-4-5  # default: the parent's model
    approval: read-only                # default: the parent's policy
    max_turns: 20
    ---
    You review code changes. ...
"""

from __future__ import annotations

import glob
import os
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from argus.skills import parse_frontmatter, read_file
from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj

if TYPE_CHECKING:
    from argus.agent import Agent
    from argus.executors import Executor

STRICTNESS = ("read-only", "ask", "auto", "full")

EXPLORE_PROMPT = (
    "You are a read-only research subagent. Find what was asked by searching and reading "
    "the code; you cannot change files. Finish with a concise report: the answer, the "
    "relevant files with line numbers, and anything surprising. No preamble."
)
GENERAL_PROMPT = (
    "You are a subagent handling one self-contained subtask for another agent. Do the "
    "task, then finish with a short report of what you did and what the other agent needs "
    "to know (files changed, commands run, open problems)."
)


@dataclass
class AgentDef:
    name: str
    description: str
    prompt: str = ""
    tools: list[str] | None = None  # None: the parent's tools
    model: str = ""  # model spec; "" = the parent's
    approval: str = ""  # "" = the parent's; never looser than it
    max_turns: int = 0
    source: str = "builtin"


BUILTIN = [
    AgentDef(
        "explore",
        "read-only research: find code, read it, answer questions about it",
        EXPLORE_PROMPT,
        tools=["read", "glob", "grep", "bash"],
        approval="read-only",
        max_turns=25,
    ),
    AgentDef(
        "general",
        "a self-contained subtask in a fresh context (can edit files); give it all it needs",
        GENERAL_PROMPT,
        max_turns=40,
    ),
]


def stricter(a: str, b: str) -> str:
    """The stricter of two approval policies ("" means no constraint)."""
    if not a:
        return b
    if not b:
        return a
    return a if STRICTNESS.index(a) <= STRICTNESS.index(b) else b


def _parse(text: str, source: str, fallback_name: str) -> AgentDef | None:
    meta, body = parse_frontmatter(text)
    name = (meta.get("name") or fallback_name).strip()
    desc = " ".join((meta.get("description") or "").split())
    if not name or not desc:
        return None
    tools = None
    if meta.get("tools"):
        tools = [t.strip() for t in meta["tools"].replace(";", ",").split(",") if t.strip()]
    approval = meta.get("approval", "").strip()
    if approval and approval not in STRICTNESS:
        approval = "read-only"
    try:
        max_turns = int(meta.get("max_turns") or 0)
    except ValueError:
        max_turns = 0
    return AgentDef(
        name, desc, body.strip(), tools, meta.get("model", "").strip(), approval, max_turns, source
    )


def discover(executor: Executor, dirs: list[str]) -> list[AgentDef]:
    """Built-in agents plus ``*.md`` definitions; a later definition replaces a built-in."""
    defs = {d.name: d for d in BUILTIN}
    for d in dirs:
        if d.startswith(("~", "local:")):
            root = os.path.expanduser(d.removeprefix("local:"))
            paths = [(p, True) for p in sorted(glob.glob(os.path.join(root, "*.md")))]
        else:
            q = shlex.quote(executor.resolve(d))
            r = executor.run(
                f'for f in {q}/*.md; do [ -f "$f" ] && printf "%s\\0" "$f"; done; true',
                timeout=30,
                merge_stderr=False,
            )
            paths = [(p, False) for p in sorted(r.output.split("\0")) if p]
        for path, local in paths:
            try:
                text = read_file(executor, path, local)
            except OSError:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            parsed = _parse(text, path, stem)
            if parsed is not None:
                defs[parsed.name] = parsed
    return list(defs.values())


class TaskTool(Tool):
    name = "task"
    summary = "hand a self-contained subtask to a subagent (fresh context); returns its report"
    mutating = True  # a subagent may edit; the parent checkpoints around it

    def __init__(self, parent: Agent, defs: list[AgentDef]):
        self.parent = parent
        self.defs = {d.name: d for d in defs}
        listing = "; ".join(f"{d.name}: {d.description}" for d in defs)
        self.description = (
            "Start a subagent with a fresh context for a self-contained subtask and get back "
            f"its final report. Agents: {listing}. The prompt must contain everything it needs."
        )
        self.parameters = obj(
            {
                "agent": {"type": "string", "enum": list(self.defs)},
                "prompt": {"type": "string"},
            },
            ["agent", "prompt"],
        )

    def signature(self) -> str:
        names = "|".join(self.defs)
        return f"task(agent: {names}, prompt) - {self.summary}"

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        d = self.defs.get(args["agent"])
        if d is None:
            raise ToolError(f"no agent {args['agent']!r}; available: {', '.join(self.defs)}")
        if not str(args.get("prompt", "")).strip():
            raise ToolError("give the subagent a prompt with everything it needs")
        r = self.parent.spawn(d, args["prompt"], ctx.run_id, ctx.turn)
        answer = (
            r.final.strip()
            or f"(no report; the subagent ended {r.status}: {r.error or ', '.join(r.tags) or 'no answer'})"
        )
        stats = (
            f"[subagent {d.name}: {r.status}, {r.turns} turns, {r.tool_calls} tool calls, "
            f"{r.completion_tokens} tokens generated, run {r.run_id}]"
        )
        return ToolResult(
            f"{answer}\n{stats}",
            ok=r.status == "completed",
            error=None if r.status == "completed" else f"subagent {r.status}",
            meta={"agent": d.name, "run": r.run_id, "status": r.status, "cost_usd": r.cost_usd},
        )
