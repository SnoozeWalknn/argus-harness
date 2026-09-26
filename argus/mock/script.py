"""Scripts for the mock server.

A script is a list of *steps*; each chat request consumes the next one. A step
describes what the model does; the server renders it for whatever protocol the
request uses. Step keys (all optional):

``reasoning``         reasoning text (sent as ``reasoning_content``)
``reasoning_repeat``  repeat the reasoning text N times (runaway reasoning)
``content``           visible text
``content_repeat``    repeat the content N times (runaway output)
``tool_calls``        ``[{"name": str, "arguments": dict | str, "id"?: str}]``;
                      a string ``arguments`` is sent verbatim (malformed JSON)
``thought``           value for the envelope's ``thought`` field (constrained protocols)
``raw``               for constrained protocols: exact content, bypassing the envelope
``finish_reason``     override
``error``             ``{"status": int, "message": str, "type": str}`` HTTP error
``delay``             seconds to wait before responding
``chunk_delay``       seconds between streamed chunks
``expect``            request assertions (see :func:`check_expect`)
``times``             repeat this step N times (loops)

A step may also be a callable ``(request, server) -> step`` for dynamic tests.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

Step = dict[str, Any]
StepFn = Callable[[dict[str, Any], Any], Step]


def call(name: str, /, **args: Any) -> Step:
    return {"tool_calls": [{"name": name, "arguments": args}]}


def calls(*pairs: tuple[str, dict[str, Any]]) -> Step:
    return {"tool_calls": [{"name": n, "arguments": a} for n, a in pairs]}


def raw_call(name: str, raw_arguments: str) -> Step:
    return {"tool_calls": [{"name": name, "arguments": raw_arguments}]}


def final(text: str = "Done.") -> Step:
    return {"content": text}


def with_(step: Step, **extra: Any) -> Step:
    return {**step, **extra}


def expand(steps: list[Any]) -> list[Any]:
    out: list[Any] = []
    for s in steps:
        if isinstance(s, dict) and "times" in s:
            n = int(s["times"])
            body = {k: v for k, v in s.items() if k != "times"}
            out.extend(dict(body) for _ in range(n))
        else:
            out.append(s)
    return out


class Script:
    def __init__(self, steps: list[Step | StepFn] | None = None, on_exhausted: str = "error"):
        if on_exhausted not in ("error", "final", "repeat_last"):
            raise ValueError("on_exhausted must be error, final or repeat_last")
        self.steps = expand(list(steps or []))
        self.on_exhausted = on_exhausted
        self.index = 0
        self._lock = threading.Lock()

    @classmethod
    def always(cls, step: Step | StepFn) -> Script:
        return cls([step], on_exhausted="repeat_last")

    @classmethod
    def load(cls, path: str | Path) -> Script:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            return cls(data)
        return cls(data.get("steps", []), data.get("on_exhausted", "error"))

    @property
    def remaining(self) -> int:
        return max(len(self.steps) - self.index, 0)

    def next(self, request: dict[str, Any], server: Any) -> Step:
        with self._lock:
            if self.index < len(self.steps):
                step = self.steps[self.index]
                self.index += 1
            elif self.on_exhausted == "repeat_last" and self.steps:
                step = self.steps[-1]
            elif self.on_exhausted == "final":
                step = final("Done.")
            else:
                return {
                    "error": {
                        "status": 500,
                        "message": "mock script exhausted",
                        "type": "mock_error",
                    }
                }
        if callable(step):
            step = step(request, server)
        return dict(step)


def _messages_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def request_protocol(req: dict[str, Any]) -> str:
    if req.get("grammar"):
        return "grammar"
    rf = req.get("response_format") or {}
    if rf.get("type") in ("json_schema", "json_object") or req.get("json_schema"):
        return "json_schema"
    return "native"


def check_expect(expect: dict[str, Any], req: dict[str, Any]) -> list[str]:
    """Assertions a step can make about the request that consumed it."""
    msgs = req.get("messages") or []
    last = msgs[-1] if msgs else {}
    problems = []
    for key, want in expect.items():
        if key == "last_role":
            if last.get("role") != want:
                problems.append(f"last message role {last.get('role')!r} != {want!r}")
        elif key == "last_contains":
            if want not in _messages_text(last):
                problems.append(f"last message lacks {want!r}: {_messages_text(last)[:200]!r}")
        elif key == "any_contains":
            if not any(want in _messages_text(m) for m in msgs):
                problems.append(f"no message contains {want!r}")
        elif key == "none_contains":
            if any(want in _messages_text(m) for m in msgs):
                problems.append(f"a message contains {want!r}")
        elif key == "system_contains":
            if not msgs or want not in _messages_text(msgs[0]):
                problems.append(f"system prompt lacks {want!r}")
        elif key == "protocol":
            if request_protocol(req) != want:
                problems.append(f"protocol {request_protocol(req)!r} != {want!r}")
        elif key == "tools":
            names = [t["function"]["name"] for t in req.get("tools") or []]
            if sorted(names) != sorted(want):
                problems.append(f"tools {names} != {want}")
        elif key == "has":
            problems += [f"request lacks key {k!r}" for k in want if k not in req]
        elif key == "lacks":
            problems += [f"request has key {k!r}" for k in want if k in req]
        elif key == "n_messages":
            if len(msgs) != want:
                problems.append(f"{len(msgs)} messages != {want}")
        elif key == "max_messages":
            if len(msgs) > want:
                problems.append(f"{len(msgs)} messages > {want}")
        elif key in req:
            if req[key] != want:
                problems.append(f"{key}={req[key]!r} != {want!r}")
        else:
            problems.append(f"unknown expectation {key!r}")
    return problems
