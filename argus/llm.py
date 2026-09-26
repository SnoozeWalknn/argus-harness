"""OpenAI-compatible client for llama-server.

Streams SSE by default and assembles ``reasoning_content``, ``content`` and
``tool_calls`` deltas into a :class:`Completion`. Stream *monitors* see every
delta and may raise :class:`StopGeneration`; the connection is then closed,
which makes llama-server stop generating, and the partial completion is
returned with ``aborted`` set.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.type = ""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                self.type = str(err.get("type", ""))


class ContextOverflow(LLMError):
    """The prompt does not fit in the server's context window."""


class StopGeneration(Exception):
    """Raised by a stream monitor to abort the current generation."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class RawToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[RawToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float | None = None
    total_ms: float = 0.0
    aborted: str | None = None  # monitor reason when generation was cut short
    abort_detail: str = ""
    reasoning_chunks: int = 0
    content_chunks: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or self.timings.get("prompt_n") or 0)

    @property
    def completion_tokens(self) -> int:
        n = self.usage.get("completion_tokens") or self.timings.get("predicted_n")
        if n:
            return int(n)
        # Aborted streams never receive usage; one chunk is ~one token in llama-server.
        return self.reasoning_chunks + self.content_chunks

    @property
    def cached_tokens(self) -> int:
        details = self.usage.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or self.timings.get("cache_n") or 0)


@dataclass
class Delta:
    kind: str  # "reasoning" | "content" | "tool"
    text: str
    completion: Completion  # the partial completion so far


Monitor = Callable[[Delta], None]


def root_url(base_url: str) -> str:
    """llama-server's native endpoints (/tokenize, /props…) live at the root, not under /v1."""
    u = base_url.rstrip("/")
    return u[: -len("/v1")] if u.endswith("/v1") else u


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 600.0,
        connect_timeout: float = 5.0,
        retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.root = root_url(self.base_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.retries = retries
        self.http = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            trust_env=False,  # never route localhost traffic through a proxy
        )

    def close(self) -> None:
        self.http.close()

    # -- chat ---------------------------------------------------------------------------

    def chat(
        self,
        body: dict[str, Any],
        *,
        stream: bool = True,
        monitors: Iterable[Monitor] = (),
    ) -> Completion:
        monitors = list(monitors)
        attempt = 0
        while True:
            try:
                if stream:
                    return self._chat_stream(body, monitors)
                return self._chat_once(body)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                # Only reached before any output was produced (the stream path
                # re-raises mid-stream failures as LLMError).
                attempt += 1
                if attempt > self.retries:
                    raise LLMError(f"cannot reach {self.base_url}: {e}") from e
                time.sleep(min(0.5 * 2**attempt, 8))
            except httpx.TimeoutException as e:
                raise LLMError(f"timed out waiting for {self.base_url}: {e!r}") from e
            except httpx.HTTPError as e:
                raise LLMError(f"request to {self.base_url} failed: {e!r}") from e
            except json.JSONDecodeError as e:
                raise LLMError(f"invalid JSON from {self.base_url}: {e}") from e

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        msg = body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            msg = body["error"].get("message", body)
        err_cls = LLMError
        if isinstance(body, dict):
            err = body.get("error") or {}
            text = f"{err.get('type', '')} {err.get('message', '')}".lower()
            if "exceed_context" in text or "context size" in text or "context length" in text:
                err_cls = ContextOverflow
        raise err_cls(f"HTTP {resp.status_code}: {msg}", status=resp.status_code, body=body)

    def _chat_once(self, body: dict[str, Any]) -> Completion:
        t0 = time.perf_counter()
        resp = self.http.post(self._url(), json={**body, "stream": False})
        self._raise_for(resp)
        data = resp.json()
        c = Completion(raw=data)
        c.total_ms = (time.perf_counter() - t0) * 1000
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        c.content = msg.get("content") or ""
        c.reasoning = msg.get("reasoning_content") or ""
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            c.tool_calls.append(RawToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        c.finish_reason = choice.get("finish_reason")
        c.usage = data.get("usage") or {}
        c.timings = data.get("timings") or {}
        return c

    def _chat_stream(self, body: dict[str, Any], monitors: list[Monitor]) -> Completion:
        req = {**body, "stream": True, "stream_options": {"include_usage": True}}
        c = Completion()
        calls: dict[int, dict[str, str]] = {}
        t0 = time.perf_counter()
        got_data = False
        try:
            with self.http.stream("POST", self._url(), json=req) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for(resp)
                try:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("error:"):
                            raise LLMError(f"stream error: {line[6:].strip()}")
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        got_data = True
                        if "error" in chunk:
                            raise LLMError(f"stream error: {chunk['error']}", body=chunk)
                        self._apply_chunk(c, calls, chunk, t0, monitors)
                except StopGeneration as stop:
                    c.aborted = stop.reason
                    c.abort_detail = stop.detail
                except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                    if not got_data:
                        raise
                    raise LLMError(f"stream interrupted: {e}") from e
        except httpx.ReadTimeout as e:
            if not got_data:
                raise
            raise LLMError(f"stream stalled (no data for {self.http.timeout.read}s): {e!r}") from e
        c.total_ms = (time.perf_counter() - t0) * 1000
        for idx in sorted(calls):
            tc = calls[idx]
            c.tool_calls.append(RawToolCall(tc["id"] or f"call_{idx}", tc["name"], tc["arguments"]))
        c.raw = {
            "stream": True,
            "message": {
                "content": c.content,
                "reasoning_content": c.reasoning,
                "tool_calls": [tc.__dict__ for tc in c.tool_calls],
            },
            "finish_reason": c.finish_reason,
            "usage": c.usage,
            "timings": c.timings,
            "aborted": c.aborted,
        }
        return c

    @staticmethod
    def _apply_chunk(
        c: Completion,
        calls: dict[int, dict[str, str]],
        chunk: dict[str, Any],
        t0: float,
        monitors: list[Monitor],
    ) -> None:
        if chunk.get("usage"):
            c.usage = chunk["usage"]
        if chunk.get("timings"):
            c.timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                c.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            events: list[tuple[str, str]] = []
            if delta.get("reasoning_content"):
                c.reasoning += delta["reasoning_content"]
                c.reasoning_chunks += 1
                events.append(("reasoning", delta["reasoning_content"]))
            if delta.get("content"):
                c.content += delta["content"]
                c.content_chunks += 1
                events.append(("content", delta["content"]))
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(calls))
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
                c.content_chunks += 1
                events.append(("tool", fn.get("arguments") or fn.get("name") or ""))
            if events and c.ttft_ms is None:
                c.ttft_ms = (time.perf_counter() - t0) * 1000
            for kind, text in events:
                d = Delta(kind, text, c)
                for m in monitors:
                    m(d)

    # -- llama-server native endpoints --------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        resp = self.http.post(f"{self.root}/tokenize", json={"content": text, "add_special": False})
        self._raise_for(resp)
        return list(resp.json().get("tokens") or [])

    def apply_template(self, body: dict[str, Any]) -> str:
        resp = self.http.post(f"{self.root}/apply-template", json=body)
        self._raise_for(resp)
        return resp.json()["prompt"]

    def props(self) -> dict[str, Any]:
        resp = self.http.get(f"{self.root}/props")
        self._raise_for(resp)
        return resp.json()

    def context_window(self) -> int:
        """n_ctx per slot as reported by /props, or 0 if unknown."""
        try:
            p = self.props()
        except (httpx.HTTPError, LLMError, ValueError):
            return 0
        dgs = p.get("default_generation_settings") or {}
        return int(dgs.get("n_ctx") or p.get("n_ctx") or 0)

    def health(self) -> bool:
        try:
            return self.http.get(f"{self.root}/health").status_code == 200
        except httpx.HTTPError:
            return False
