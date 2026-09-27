"""Approval policies: which tool calls run, sandboxed how, and when to ask a human.

========== ===================== =========================================== ======================
policy     ``edit``              ``bash``                                    escalation
========== ===================== =========================================== ======================
read-only  denied                read-only sandbox, no network               none
ask        asks every time       asks (known-safe read-only commands run in  –
                                 the read-only sandbox without asking); runs
                                 in the workspace-write sandbox
auto       allowed in the        workspace-write sandbox                     a failure that looks
           workspace, asks                                                   like a sandbox denial
           outside it                                                        asks to re-run
                                                                             unsandboxed
full       allowed               unsandboxed                                 –
========== ===================== =========================================== ======================

Without a working sandbox, ``auto`` runs commands unsandboxed with a warning
(unless ``sandbox.required``), and ``read-only`` / ``ask`` cannot offer a
read-only sandbox, so read-only commands are denied or asked about.

"Asking" goes to an :class:`Approver`: a terminal prompt, the TUI, or the
headless default that answers from ``agent.headless_approval``. An "always"
answer is remembered for the rest of the session: per tool for edits, per
command prefix for bash.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from argus.sandbox import OFF, READ_ONLY, WORKSPACE_WRITE, Backend

POLICIES = ("read-only", "ask", "auto", "full")
ANSWERS = ("yes", "no", "always")

DENIAL = re.compile(
    r"Read-only file system|Permission denied|Operation not permitted|Network is unreachable"
    r"|Temporary failure in name resolution|Could not resolve host|Name or service not known"
    r"|getaddrinfo failed|\[Errno 13\]|\[Errno 30\]|\[Errno 101\]",
    re.I,
)

SAFE = {
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "pwd", "echo", "true",
    "false", "which", "whoami", "id", "uname", "date", "stat", "file", "tree", "du", "df",
    "sort", "uniq", "cut", "tr", "nl", "rev", "seq", "diff", "cmp", "basename", "dirname",
    "realpath", "readlink", "printf", "test", "[",
}  # fmt: skip
SAFE_GIT = {"status", "log", "diff", "show", "branch", "blame", "ls-files", "rev-parse", "grep"}
UNSAFE_FIND = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf"}
SEPARATORS = {";", "&&", "||", "|"}


def is_known_safe(cmd: str) -> bool:
    """A command that only reads: every segment of a simple pipeline is on the allow list."""
    if any(s in cmd for s in ("$(", "`", "<(", ">(")):
        return False
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return False
    if not tokens:
        return False
    segments: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in SEPARATORS:
            segments.append([])
        elif t and set(t) <= set("<>&"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if (t == ">&" and nxt == "1") or (t == ">" and nxt == "/dev/null"):
                if segments[-1] and segments[-1][-1] == "2":  # the fd of 2>&1, 2>/dev/null
                    segments[-1].pop()
                i += 1
            else:
                return False  # other redirections could write files; & runs in background
        else:
            segments[-1].append(t)
        i += 1
    return all(seg and _safe_segment(seg) for seg in segments)


def _safe_segment(seg: list[str]) -> bool:
    prog = seg[0].rsplit("/", 1)[-1]
    if prog in SAFE:
        return True
    if prog == "git":
        sub = next((a for a in seg[1:] if not a.startswith("-")), "")
        return sub in SAFE_GIT
    if prog == "find":
        return not UNSAFE_FIND & set(seg)
    if prog == "sed":
        return "-n" in seg and not any(a.startswith("-i") for a in seg)
    return False


def command_prefix(cmd: str) -> str:
    """What an "always" answer covers: the program and its subcommand, e.g. ``npm test``."""
    try:
        words = shlex.split(cmd)
    except ValueError:
        words = cmd.split()
    words = [w for w in words if "=" not in w or w.startswith("-")][:3]
    if not words:
        return cmd.strip()
    if words[0] in ("python", "python3", "uv", "npx", "npm", "pnpm", "yarn", "cargo", "go", "git"):
        return " ".join(words[:2] if len(words) > 1 and words[1] != "-m" else words[:3])
    return words[0]


@dataclass
class Request:
    tool: str
    summary: str  # e.g. "bash: pytest -q"
    reason: str  # why argus asks
    args: dict[str, Any] = field(default_factory=dict)


class Approver:
    """Answers "yes", "no" or "always"."""

    interactive = False

    def ask(self, req: Request) -> str:
        raise NotImplementedError


class FixedApprover(Approver):
    def __init__(self, answer: str = "no"):
        self.answer = answer

    def ask(self, req: Request) -> str:
        return self.answer


class CallbackApprover(Approver):
    interactive = True

    def __init__(self, fn: Callable[[Request], str]):
        self.fn = fn

    def ask(self, req: Request) -> str:
        return self.fn(req)


class TTYApprover(Approver):
    """Asks on the terminal: y(es) / n(o) / a(lways)."""

    interactive = True

    def __init__(self, stream: Any = None, read: Callable[[], str] | None = None):
        self.stream = stream or sys.stderr
        self.read = read or (lambda: input())

    def ask(self, req: Request) -> str:
        self.stream.write(
            f"\n  ? {req.summary}\n    ({req.reason})  allow? [y]es / [n]o / [a]lways: "
        )
        self.stream.flush()
        try:
            answer = self.read().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "no"
        return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(answer, "no")


@dataclass
class Decision:
    allow: bool
    sandbox: str = OFF  # read-only | workspace-write | off
    reason: str = ""
    source: str = "policy"  # policy | safe-command | approver | session | no-sandbox
    asked: bool = False
    answer: str = ""


class Gate:
    def __init__(
        self,
        policy: str,
        backend: Backend,
        approver: Approver,
        *,
        can_sandbox: bool = True,
        required: bool = False,
    ):
        if policy not in POLICIES:
            raise ValueError(f"approval policy must be one of {POLICIES}, got {policy!r}")
        self.policy = policy
        self.backend = backend
        self.approver = approver
        self.sandboxed = can_sandbox and backend.name != "none"
        self.required = required
        self.always: set[str] = set()  # remembered "always" answers
        self.warned = False

    # -- decisions ---------------------------------------------------------------------------

    def decide(
        self,
        tool: str,
        args: dict[str, Any],
        mutating: bool,
        inside: bool,
        approver: Approver | None = None,
    ) -> Decision:
        """``inside``: whether a file tool's path is inside the workspace. ``approver``
        answers instead of the gate's own (a pre_tool hook that already allowed the call)."""
        if approver is not None:
            saved, self.approver = self.approver, approver
            try:
                return self.decide(tool, args, mutating, inside)
            finally:
                self.approver = saved
        if tool == "bash":
            return self._bash(args.get("cmd", ""))
        if tool == "task":  # subagents run under a policy no looser than this one
            return Decision(True)
        if not mutating:
            return Decision(True)
        p = self.policy
        if p == "full":
            return Decision(True)
        if p == "read-only":
            return Decision(False, reason="the read-only policy does not allow file changes")
        if p == "auto" and inside:
            return Decision(True)
        why = "edit outside the workspace" if not inside else "the ask policy confirms each edit"
        key = f"{tool}:{'outside' if not inside else 'inside'}"
        return self._ask(key, Request(tool, f"{tool}: {args.get('path', '')}", why, args), OFF)

    def _bash(self, cmd: str) -> Decision:
        p = self.policy
        if p == "full":
            return Decision(True)
        if not self.sandboxed:
            if self.required or p == "read-only":
                why = "no sandbox is available to run commands" + (
                    " read-only" if p == "read-only" else ""
                )
                return Decision(False, reason=why, source="no-sandbox")
            if p == "auto":
                return Decision(True, OFF, "no sandbox available", "no-sandbox")
        if p == "read-only":
            return Decision(True, READ_ONLY)
        if p == "auto":
            return Decision(True, WORKSPACE_WRITE)
        # ask
        if is_known_safe(cmd):
            return Decision(True, READ_ONLY if self.sandboxed else OFF, source="safe-command")
        mode = WORKSPACE_WRITE if self.sandboxed else OFF
        req = Request("bash", f"bash: {cmd}", "the ask policy confirms each command", {"cmd": cmd})
        return self._ask(f"bash:{command_prefix(cmd)}", req, mode)

    def _ask(self, key: str, req: Request, sandbox: str) -> Decision:
        if key in self.always:
            return Decision(True, sandbox, req.reason, "session")
        answer = self.approver.ask(req)
        if answer == "always":
            self.always.add(key)
        allowed = answer in ("yes", "always")
        reason = req.reason if allowed else f"not approved ({req.reason})"
        return Decision(allowed, sandbox, reason, "approver", asked=True, answer=answer)

    def escalate(self, cmd: str) -> Decision | None:
        """After a sandboxed command failed like a sandbox denial: may it run unsandboxed?"""
        if self.policy not in ("auto", "ask"):
            return None
        req = Request(
            "bash",
            f"bash (unsandboxed): {cmd}",
            "it failed inside the sandbox, apparently blocked by it",
            {"cmd": cmd},
        )
        return self._ask(f"escalate:{command_prefix(cmd)}", req, OFF)


def looks_denied(output: str) -> bool:
    return bool(DENIAL.search(output or ""))
