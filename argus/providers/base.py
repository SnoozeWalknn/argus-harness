"""What every model API adapter shares: result types, errors, HTTP and SSE plumbing.

An adapter turns argus's canonical request (OpenAI chat-completions messages plus
a few neutral knobs, see :mod:`argus.providers`) into its API's wire format and
the response back into a :class:`Completion`. Streaming adapters emit the same
:class:`Delta` events, so stream monitors can abort any provider's generation
by raising :class:`StopGeneration`; the connection is then closed and the
partial completion returned with ``aborted`` set.
"""

from __future__ import annotations

import ipaddress
import json
import math
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from argus.providers.record import Recorder

CHARS_PER_TOKEN = 3.7  # used where an API streams several tokens per chunk


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: Any = None,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after
        self.type = ""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                self.type = str(err.get("type") or err.get("status") or err.get("code") or "")


class ContextOverflow(LLMError):
    """The prompt does not fit in the model's context window."""


class NotSupported(LLMError):
    """The provider has no such endpoint (e.g. /tokenize outside llama-server)."""


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
    finish_reason: str | None = None  # stop | tool_calls | length | refusal
    # Normalised usage: prompt_tokens (all input, cached included), completion_tokens,
    # prompt_tokens_details.cached_tokens, completion_tokens_details.reasoning_tokens,
    # and cache_write_tokens where the provider bills cache writes.
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float | None = None
    total_ms: float = 0.0
    aborted: str | None = None  # monitor reason when generation was cut short
    abort_detail: str = ""
    reasoning_chunks: int = 0
    content_chunks: int = 0
    token_chunks: bool = True  # one streamed chunk is one token (llama.cpp and friends)
    stop_raw: str | None = None  # the provider's own stop reason
    refusal: str | None = None  # refusal text or category, when the model declined
    malformed: str | None = None  # the provider reports an unparseable tool call
    replay: dict[str, Any] | None = None  # provider state to send back with this message
    notes: list[str] = field(default_factory=list)  # adapter decisions worth logging
    replay_rejected: bool = False  # the API refused replayed state; stop sending it
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or self.timings.get("prompt_n") or 0)

    @property
    def completion_tokens(self) -> int:
        n = self.usage.get("completion_tokens") or self.timings.get("predicted_n")
        if n:
            return int(n)
        # Aborted streams never receive usage.
        if self.token_chunks:
            return self.reasoning_chunks + self.content_chunks
        text = self.reasoning + self.content + "".join(t.arguments for t in self.tool_calls)
        return estimate_tokens(text)

    @property
    def cached_tokens(self) -> int:
        details = self.usage.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or self.timings.get("cache_n") or 0)

    @property
    def cache_write_tokens(self) -> int:
        return int(self.usage.get("cache_write_tokens") or 0)

    @property
    def reported_reasoning_tokens(self) -> int | None:
        details = self.usage.get("completion_tokens_details") or {}
        n = details.get("reasoning_tokens")
        return int(n) if n is not None else None

    @property
    def reasoning_tokens(self) -> int:
        """Reasoning tokens so far: chunks where a chunk is a token, else an estimate."""
        if self.token_chunks:
            return self.reasoning_chunks
        return estimate_tokens(self.reasoning)


@dataclass
class Delta:
    kind: str  # "reasoning" | "content" | "tool"
    text: str
    completion: Completion  # the partial completion so far


Monitor = Callable[[Delta], None]


@dataclass
class Detected:
    """What a server or API says about a model."""

    flavor: str = ""
    context_window: int = 0
    max_output: int = 0
    models: list[str] = field(default_factory=list)
    capabilities: set[str] = field(default_factory=set)  # tools, thinking, vision, ...
    pricing: dict[str, float] = field(default_factory=dict)  # USD per 1M tokens
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderOptions:
    """Per-model adjustments, normally filled from the model's profile."""

    flavor: str = (
        "auto"  # openai_compat: llama_server | ollama | vllm | lmstudio | openrouter | generic
    )
    quirks: frozenset[str] = frozenset()
    effort: str = ""  # reasoning effort, where the API has one
    thinking_budget: int = 0  # reasoning token budget, where the API takes one
    cache: bool = True  # prompt caching, where the API needs it requested
    thinking: str = ""  # thinking format: "" = the adapter's default, or e.g. adaptive | budget
    headers: dict[str, str] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def root_url(base_url: str) -> str:
    """llama-server's native endpoints (/tokenize, /props…) live at the root, not under /v1."""
    u = base_url.rstrip("/")
    return u[: -len("/v1")] if u.endswith("/v1") else u


def is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


@dataclass
class SSE:
    event: str | None
    data: str

    def json(self) -> Any:
        return json.loads(self.data)


def iter_sse(lines: Iterable[str]) -> Iterator[SSE]:
    """Server-sent events: ``event:`` and ``data:`` fields, dispatched on a blank line.

    Comment lines (``: keep-alive``) are skipped. llama-server reports errors in a
    stream as a bare ``error: {...}`` line, which becomes an ``error`` event.
    """
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield SSE(event, "\n".join(data))
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
        elif name == "error" and not data:
            yield SSE("error", value)
    if data:
        yield SSE(event, "\n".join(data))


class Provider:
    """Base adapter: an httpx client, retries, error mapping and recording."""

    kind = "base"
    default_base_url = ""
    token_chunks = False
    retry_statuses: tuple[int, ...] = (408, 429, 503, 529)

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        timeout: float = 600.0,
        connect_timeout: float = 5.0,
        retries: int = 2,
        *,
        model: str = "",
        options: ProviderOptions | None = None,
        recorder: Recorder | None = None,
        max_retry_wait: float = 60.0,
    ):
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.api_key = api_key
        self.model = model
        self.opts = options or ProviderOptions()
        self.retries = retries
        self.max_retry_wait = max_retry_wait
        self.recorder = recorder
        self._detected: Detected | None = None
        self.http = httpx.Client(
            headers={**self.auth_headers(), **self.opts.headers},
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            # Local servers never go through a proxy; cloud APIs use the environment's.
            trust_env=not is_loopback(self.base_url),
        )

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def strip_replay(replay: dict[str, Any]) -> dict[str, Any] | None:
        """Replay state without its reasoning, for history that was edited.

        What remains (e.g. the exact tool-call blocks) keeps later requests identical
        to the one the provider last accepted.
        """
        return None

    def close(self) -> None:
        self.http.close()

    def describe(self) -> str:
        return f"{self.kind} {self.base_url}"

    # -- the interface ---------------------------------------------------------------------

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
                return self._attempt(body, stream, monitors)
            except LLMError as e:
                if not e.retryable or attempt >= self.retries:
                    raise
                attempt += 1
                self._sleep(e.retry_after, attempt)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                # Only reached before any output was produced (the stream path
                # re-raises mid-stream failures as LLMError).
                attempt += 1
                if attempt > self.retries:
                    raise LLMError(f"cannot reach {self.base_url}: {e}") from e
                self._sleep(None, attempt)
            except httpx.TimeoutException as e:
                raise LLMError(f"timed out waiting for {self.base_url}: {e!r}") from e
            except httpx.HTTPError as e:
                raise LLMError(f"request to {self.base_url} failed: {e!r}") from e
            except json.JSONDecodeError as e:
                raise LLMError(f"invalid JSON from {self.base_url}: {e}") from e

    def _sleep(self, hint: float | None, attempt: int) -> None:
        wait = hint if hint is not None else min(0.5 * 2**attempt, 8.0)
        time.sleep(min(wait, self.max_retry_wait))

    def _attempt(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        c = self._chat(body, stream, monitors)
        c.provider = c.provider or self.kind
        c.model = c.model or str(body.get("model") or self.model)
        return c

    def _chat(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        raise NotImplementedError

    def count_prompt(self, body: dict[str, Any]) -> int | None:
        """Exact prompt tokens of a request, where the API can count them."""
        return None

    def tokenize(self, text: str) -> list[int]:
        raise NotSupported(f"{self.kind} has no tokenizer endpoint")

    def apply_template(self, body: dict[str, Any]) -> str:
        raise NotSupported(f"{self.kind} does not expose its chat template")

    def detect(self, refresh: bool = False) -> Detected:
        if self._detected is None or refresh:
            try:
                self._detected = self._detect()
            except (httpx.HTTPError, LLMError, ValueError, KeyError, TypeError) as e:
                self._detected = Detected(notes=[f"detection failed: {type(e).__name__}: {e}"])
        return self._detected

    def _detect(self) -> Detected:
        return Detected()

    def context_window(self) -> int:
        """The model's context window as the server reports it, or 0 if unknown."""
        return self.detect().context_window

    def health(self) -> bool:
        return bool(self.detect().models or self.detect().context_window)

    # -- HTTP helpers ------------------------------------------------------------------------

    def error_class(self, status: int, message: str, body: Any) -> type[LLMError]:
        text = message.lower()
        if any(
            k in text
            for k in (
                "exceed_context",
                "context size",
                "context length",
                "context_length_exceeded",
                "maximum context length",
                "context window",
                "prompt is too long",
                "input token count",
                "too many tokens",
            )
        ):
            return ContextOverflow
        return LLMError

    def raise_for(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        msg: Any = body
        kind = ""
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            msg = err.get("message", body)
            kind = f"{err.get('type') or err.get('status') or ''} {err.get('code') or ''}"
        elif isinstance(body, dict) and "message" in body:
            msg = body["message"]
        text = f"{kind} {msg}"
        cls = self.error_class(resp.status_code, text, body)
        retryable = cls is LLMError and resp.status_code in self.retry_statuses
        raise cls(
            f"HTTP {resp.status_code}: {msg}",
            status=resp.status_code,
            body=body,
            retryable=retryable,
            retry_after=retry_after(resp),
        )

    def post_json(self, url: str, payload: dict[str, Any], ex: Any = None) -> Any:
        resp = self.http.post(url, json=payload)
        if ex is not None:
            ex.response_json(resp)
        self.raise_for(resp)
        return resp.json()

    @contextmanager
    def stream_events(self, url: str, payload: dict[str, Any], ex: Any = None) -> Iterator[Any]:
        """POST and iterate SSE events; HTTP errors are raised before the first event."""
        with self.http.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                resp.read()
                if ex is not None:
                    ex.response_json(resp)
                self.raise_for(resp)
            if ex is not None:
                ex.response_stream(resp)
            events = iter_sse(resp.iter_lines())
            yield ex.tee(events) if ex is not None else events

    def exchange(
        self, path: str, canonical: dict[str, Any], wire: dict[str, Any], stream: bool
    ) -> Any:
        """Recording context for one request (a no-op unless a recorder is set)."""
        from argus.providers.record import NullExchange

        if self.recorder is None:
            return NullExchange()
        return self.recorder.exchange(self, path, canonical, wire, stream)

    def get_json(self, url: str) -> Any:
        resp = self.http.get(url)
        self.raise_for(resp)
        return resp.json()


def run_monitors(c: Completion, events: list[tuple[str, str]], monitors: list[Monitor]) -> None:
    for kind, text in events:
        d = Delta(kind, text, c)
        for m in monitors:
            m(d)


def mark_ttft(c: Completion, t0: float) -> None:
    if c.ttft_ms is None:
        c.ttft_ms = (time.perf_counter() - t0) * 1000


def stream_guard(got_data: Callable[[], bool], timeout: Any) -> Any:
    """Context manager mapping transport failures after the first event to LLMError."""
    return _StreamGuard(got_data, timeout)


class _StreamGuard:
    def __init__(self, got_data: Callable[[], bool], timeout: Any):
        self.got_data = got_data
        self.timeout = timeout

    def __enter__(self) -> None:
        return None

    def __exit__(self, et: Any, e: Any, tb: Any) -> bool:
        if e is None or not self.got_data():
            return False
        if isinstance(e, httpx.ReadTimeout):
            raise LLMError(f"stream stalled (no data for {self.timeout}s): {e!r}") from e
        if isinstance(e, (httpx.ReadError, httpx.RemoteProtocolError)):
            raise LLMError(f"stream interrupted: {e}") from e
        return False


def text_of(content: Any) -> str:
    """Plain text of an OpenAI message ``content`` (a string or a list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)
