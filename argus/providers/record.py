"""Recording model API exchanges as fixtures.

With ``--record DIR`` (or ``ARGUS_RECORD=DIR``) every request an adapter sends
is saved as one JSON file: the canonical argus request, the exact wire request,
the exact response (a JSON body or the list of SSE events) and what the adapter
parsed from it. Headers are not recorded apart from a few harmless ones, and
the API key is scrubbed from everything, so recordings can be committed as
fixtures. ``tests/test_provider_fixtures.py`` replays every fixture it finds.

Fixture format (``format = 1``)::

    {"format": 1, "source": "recorded" | "documented", "provider": "anthropic",
     "flavor": "", "quirks": [], "model": "...", "stream": true,
     "canonical": {...},                      # argus's request (OpenAI-shaped)
     "request": {"method": "POST", "path": "/messages", "body": {...}},
     "response": {"status": 200, "headers": {...},
                  "json": {...} | "sse": [{"event": "...", "data": {...} | "[DONE]"}]},
     "expect": {"content": "...", "reasoning": "...", "tool_calls": [[id, name, args]],
                "finish_reason": "...", "usage": {...}, "replay": {...}},
     "error": {"type": "ContextOverflow", "message": "..."}}   # when the call raised
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from argus.providers.base import SSE, Completion, Provider

KEEP_HEADERS = ("content-type", "retry-after", "request-id", "x-request-id", "anthropic-ratelimit")


def expect_of(c: Completion) -> dict[str, Any]:
    out: dict[str, Any] = {
        "content": c.content,
        "reasoning": c.reasoning,
        "tool_calls": [[t.id, t.name, t.arguments] for t in c.tool_calls],
        "finish_reason": c.finish_reason,
        "usage": c.usage,
    }
    if c.replay:
        out["replay"] = c.replay
    if c.refusal:
        out["refusal"] = c.refusal
    if c.aborted:
        out["aborted"] = c.aborted
    return out


def _headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower().startswith(KEEP_HEADERS)}


def _data(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


class NullExchange:
    """Stands in for an exchange when nothing is being recorded."""

    completion: Completion | None = None

    def __enter__(self) -> NullExchange:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def response_json(self, resp: httpx.Response) -> None:
        pass

    def response_stream(self, resp: httpx.Response) -> None:
        pass

    def tee(self, events: Iterable[SSE]) -> Iterator[SSE]:
        return iter(events)


class Exchange(NullExchange):
    def __init__(
        self,
        recorder: Recorder,
        provider: Provider,
        path: str,
        canonical: dict[str, Any],
        wire: dict[str, Any],
        stream: bool,
    ):
        self.recorder = recorder
        self.secret = provider.api_key
        self.events: list[dict[str, Any]] = []
        self.completion = None
        self.data: dict[str, Any] = {
            "format": 1,
            "source": "recorded",
            "provider": provider.kind,
            "flavor": getattr(provider, "flavor_name", lambda: "")(),
            "quirks": sorted(provider.opts.quirks),
            "model": str(wire.get("model") or canonical.get("model") or provider.model),
            "stream": stream,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "canonical": canonical,
            "request": {"method": "POST", "path": path, "body": wire},
        }

    def response_json(self, resp: httpx.Response) -> None:
        self.data["response"] = {
            "status": resp.status_code,
            "headers": _headers(resp),
            "json": _data(resp.text),
        }

    def response_stream(self, resp: httpx.Response) -> None:
        self.data["response"] = {
            "status": resp.status_code,
            "headers": _headers(resp),
            "sse": self.events,
        }

    def tee(self, events: Iterable[SSE]) -> Iterator[SSE]:
        for ev in events:
            item: dict[str, Any] = {"data": _data(ev.data)}
            if ev.event:
                item = {"event": ev.event, **item}
            self.events.append(item)
            yield ev

    def __exit__(self, et: Any, e: Any, tb: Any) -> bool:
        if self.completion is not None:
            self.data["expect"] = expect_of(self.completion)
        if e is not None:
            self.data["error"] = {"type": type(e).__name__, "message": str(e)}
        self.recorder.save(self.data, self.secret)
        return False


class Recorder:
    def __init__(self, directory: str | Path):
        self.dir = Path(os.path.expanduser(str(directory)))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._n = itertools.count(1)
        self.saved: list[Path] = []

    def exchange(
        self,
        provider: Provider,
        path: str,
        canonical: dict[str, Any],
        wire: dict[str, Any],
        stream: bool,
    ) -> Exchange:
        return Exchange(self, provider, path, canonical, wire, stream)

    def save(self, data: dict[str, Any], secret: str = "") -> Path:
        text = json.dumps(data, indent=1, ensure_ascii=False, default=str)
        for s in {secret, *_env_secrets()}:
            if s and len(s) >= 8:
                text = text.replace(s, "<redacted>")
        name = f"{data['provider']}-{time.strftime('%Y%m%d-%H%M%S')}-{next(self._n):03d}.json"
        path = self.dir / name
        path.write_text(text + "\n")
        self.saved.append(path)
        return path


def _env_secrets() -> list[str]:
    names = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    return [os.environ.get(n, "") for n in (*names, "OPENROUTER_API_KEY")]


def recorder_from_env(directory: str = "") -> Recorder | None:
    target = directory or os.environ.get("ARGUS_RECORD", "")
    return Recorder(target) if target else None
