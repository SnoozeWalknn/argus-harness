"""Tool interface shared by all tools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from argus.config import ToolsConfig
from argus.executors import Executor


class ToolError(Exception):
    """Raised by a tool to return an error message the model can act on."""


@dataclass
class ToolResult:
    text: str  # what the model sees
    ok: bool = True
    error: str | None = None  # short error summary for the log
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


class FileTracker:
    """Remembers the content hash of each file as the model last saw it."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def mark(self, path: str, data: bytes) -> None:
        self._seen[path] = digest(data)

    def mark_digest(self, path: str, sha1: str) -> None:
        """Remember a file as seen with this content hash (e.g. from an earlier run)."""
        self._seen[path] = sha1

    def seen(self, path: str) -> bool:
        return path in self._seen

    def is_stale(self, path: str, data: bytes) -> bool:
        return path in self._seen and self._seen[path] != digest(data)

    def invalidate(self, path: str) -> None:
        """Mark a file as changed behind the model's back (kept as 'seen')."""
        if path in self._seen:
            self._seen[path] = "stale"

    def forget(self, path: str) -> None:
        self._seen.pop(path, None)


@dataclass
class ToolContext:
    executor: Executor
    cfg: ToolsConfig
    tracker: FileTracker = field(default_factory=FileTracker)
    # Side channel for things the agent should log (e.g. skill invocations).
    events: list[dict[str, Any]] = field(default_factory=list)
    sandbox: Any = None  # argus.sandbox.Spec for the call being run (bash), or None
    todos: list[dict[str, Any]] = field(default_factory=list)  # the todo tool's list
    run_id: str = ""  # the run and turn being executed (for tools that start subagents)
    turn: int = 0
    jobs: Any = None  # argus.tools.jobs.Jobs: background commands


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    summary: str = ""  # short phrase for the one-line tool list of the constrained protocols
    mutating: bool = False

    def signature(self) -> str:
        """One-line form, e.g. ``read(path, offset?, limit?) - show a file``."""
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        params = ", ".join(p if p in required else f"{p}?" for p in props)
        return f"{self.name}({params}) - {self.summary or self.description}"

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def schema(self, description: str | None = None) -> dict[str, Any]:
        """OpenAI ``tools`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description or self.description,
                "parameters": self.parameters,
            },
        }


def obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
