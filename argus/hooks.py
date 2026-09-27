"""Hooks: user commands that run at points of a run and can steer it (as in Claude Code).

Configured as ``[[hooks]]`` tables::

    [[hooks]]
    event = "pre_tool"          # session_start | user_prompt | pre_tool | post_tool | stop
    matcher = "bash|edit"       # regex on the tool name (tool events; empty = every tool)
    command = "./scripts/check-command.sh"
    timeout = 30

The event is passed as JSON on stdin (``event``, ``run_id``, ``workdir`` and the
event's fields) and in ``ARGUS_EVENT`` / ``ARGUS_RUN_ID`` / ``ARGUS_WORKDIR``.

Exit status 0 continues; stdout of ``session_start`` and ``user_prompt`` hooks is
added to the context and stdout of ``post_tool`` hooks to the tool result. Exit
status 2 blocks: ``user_prompt`` stops the run, ``pre_tool`` turns the call into an
error result carrying stderr, ``post_tool`` adds stderr to the result as feedback,
and ``stop`` makes the model continue with stderr as the next message. A
``pre_tool`` hook may also print ``{"decision": "allow" | "deny", "reason": ...}``
to answer the approval question itself. Other exit codes are logged and ignored.

Hooks run on the machine running argus (in the workspace for local runs) and are
not sandboxed: they are your code, not the model's.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

EVENTS = ("session_start", "user_prompt", "pre_tool", "post_tool", "stop")
BLOCK = 2
MAX_OUTPUT = 8000


@dataclass
class HookResult:
    command: str
    code: int | None
    stdout: str = ""
    stderr: str = ""
    ms: float = 0.0
    error: str = ""

    @property
    def blocked(self) -> bool:
        return self.code == BLOCK

    @property
    def ok(self) -> bool:
        return self.code == 0

    def decision(self) -> dict[str, Any] | None:
        """A JSON decision printed by a pre_tool hook, if any."""
        text = self.stdout.strip()
        if not (self.ok and text.startswith("{")):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return (
            data if isinstance(data, dict) and data.get("decision") in ("allow", "deny") else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "code": self.code,
            "ms": round(self.ms, 1),
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:2000],
            "error": self.error,
        }


@dataclass
class Hooks:
    configs: list[Any] = field(default_factory=list)  # argus.config.HookConfig
    workdir: str = ""
    cwd: str | None = None  # where hooks run (the workspace for local runs)

    def matching(self, event: str, tool: str = "") -> list[Any]:
        out = []
        for h in self.configs:
            if h.event != event:
                continue
            if tool and h.matcher and not re.fullmatch(h.matcher, tool):
                continue
            out.append(h)
        return out

    def run(
        self, event: str, run_id: str, payload: dict[str, Any], tool: str = ""
    ) -> list[HookResult]:
        results = []
        for h in self.matching(event, tool):
            results.append(self._one(h, event, run_id, payload))
        return results

    def _one(self, h: Any, event: str, run_id: str, payload: dict[str, Any]) -> HookResult:
        data = json.dumps(
            {"event": event, "run_id": run_id, "workdir": self.workdir, **payload}, default=str
        )
        env = {
            **os.environ,
            "ARGUS_EVENT": event,
            "ARGUS_RUN_ID": run_id,
            "ARGUS_WORKDIR": self.workdir,
        }
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                ["bash", "-c", h.command],
                input=data,
                capture_output=True,
                text=True,
                timeout=h.timeout,
                cwd=self.cwd if self.cwd and os.path.isdir(self.cwd) else None,
                env=env,
            )
            code, out, err, error = p.returncode, p.stdout, p.stderr, ""
        except subprocess.TimeoutExpired:
            code, out, err, error = None, "", "", f"timed out after {h.timeout:g}s"
        except OSError as e:
            code, out, err, error = None, "", "", str(e)
        ms = (time.perf_counter() - t0) * 1000
        return HookResult(h.command, code, out[:MAX_OUTPUT], err[:MAX_OUTPUT], ms, error)
