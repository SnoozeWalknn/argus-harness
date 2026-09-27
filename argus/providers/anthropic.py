"""Anthropic Messages API (``POST /v1/messages``), streamed or not.

Translation from argus's canonical (OpenAI-shaped) request:

* system messages → top-level ``system`` blocks; tool schemas → ``tools`` with
  ``input_schema``; ``tool_choice`` → ``{"type": "auto" | "any" | "none"}``
* assistant ``tool_calls`` → ``tool_use`` blocks; ``tool`` messages → ``tool_result``
  blocks, all results of a turn (and any feedback after them) in one user message
* an assistant message carrying Anthropic replay state is sent back as the exact
  content blocks the API returned, thinking blocks and signatures included
* thinking: adaptive (``{"type": "adaptive", "display": "summarized"}`` plus
  ``output_config.effort``) by default, ``budget_tokens`` for older models
  (``ProviderOptions.thinking = "budget"``); sampling parameters are dropped when
  thinking is on or the model rejects them (quirk ``no_sampling``)
* prompt caching: ``cache_control`` breakpoints on the last system block and on the
  last block of the newest message

Responses: ``thinking`` text becomes reasoning, ``text`` content, ``tool_use`` tool
calls; usage is normalised with cache reads and writes; stop reasons map to
``stop | tool_calls | length | refusal``. A 400 saying a thinking block's signature
belongs to a different conversation (history was edited) is retried once with
every thinking block stripped, and noted on the completion.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from argus.providers.base import (
    Completion,
    Detected,
    LLMError,
    Monitor,
    NotSupported,
    Provider,
    RawToolCall,
    StopGeneration,
    mark_ttft,
    run_monitors,
    stream_guard,
    text_of,
)

API_VERSION = "2023-06-01"
STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "refusal",
}
SAMPLING = ("temperature", "top_p", "top_k")
THINKING_BLOCKS = ("thinking", "redacted_thinking")
BAD_ID = re.compile(r"[^A-Za-z0-9_-]")


def tool_id(raw: str) -> str:
    """Tool-use ids must match ``^[a-zA-Z0-9_-]+$``; ids from other providers may not."""
    return BAD_ID.sub("_", raw or "") or "call"


def is_signature_error(e: LLMError) -> bool:
    text = str(e)
    return e.status == 400 and "signature" in text and "thinking" in text


class AnthropicProvider(Provider):
    kind = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    token_chunks = False
    retry_statuses = (408, 429, 500, 502, 503, 504, 529)

    @staticmethod
    def strip_replay(replay: dict[str, Any]) -> dict[str, Any] | None:
        blocks = [b for b in replay.get("content") or [] if b.get("type") not in THINKING_BLOCKS]
        return {**replay, "content": blocks} if blocks else None

    def auth_headers(self) -> dict[str, str]:
        headers = {"anthropic-version": API_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    # -- request -------------------------------------------------------------------------------

    def wire_body(
        self, body: dict[str, Any], stream: bool, strip_thinking: bool = False
    ) -> dict[str, Any]:
        for key in ("grammar", "response_format", "json_schema"):
            if body.get(key):
                raise LLMError(
                    f"anthropic does not take {key!r}; use the native tool-call protocol"
                )
        q = self.opts.quirks
        cache = body.get("cache", self.opts.cache)
        system_texts: list[str] = []
        messages: list[dict[str, Any]] = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role == "system":
                system_texts.append(text_of(m.get("content")))
            elif role == "assistant":
                self._add(messages, "assistant", self._assistant_blocks(m, strip_thinking))
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": tool_id(m.get("tool_call_id", "")),
                    "content": text_of(m.get("content")) or "(no output)",
                }
                self._add(messages, "user", [block])
            else:
                text = text_of(m.get("content"))
                self._add(messages, "user", [{"type": "text", "text": text or "(empty)"}])

        out: dict[str, Any] = {"model": body.get("model") or self.model}
        out["max_tokens"] = int(body.get("max_tokens") or 8192)
        if system_texts:
            system = [{"type": "text", "text": "\n\n".join(system_texts)}]
            if cache:
                system[-1]["cache_control"] = {"type": "ephemeral"}
            out["system"] = system
        if cache and messages:
            last = messages[-1]["content"][-1]
            if last.get("type") not in THINKING_BLOCKS:
                last["cache_control"] = {"type": "ephemeral"}
        out["messages"] = messages

        tools = body.get("tools") or []
        if tools:
            out["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters") or {"type": "object"},
                }
                for t in tools
            ]
            choice = body.get("tool_choice") or "auto"
            tc: dict[str, Any] = {"type": {"required": "any"}.get(choice, choice)}
            if tc["type"] == "any" and "no_forced_tool_choice" in q:
                tc["type"] = "auto"
            if body.get("parallel_tool_calls") is False and tc["type"] != "none":
                tc["disable_parallel_tool_use"] = True
            out["tool_choice"] = tc

        thinking_on = self._thinking(out, body.get("thinking") or {})
        for key in SAMPLING:
            value = body.get(key)
            if value is None or "no_sampling" in q:
                continue
            if thinking_on and key in ("temperature", "top_k"):
                continue  # not compatible with thinking
            if thinking_on and key == "top_p":
                value = max(float(value), 0.95)
            out[key] = value
        if stream:
            out["stream"] = True
        return out

    def _thinking(self, out: dict[str, Any], thinking: dict[str, Any]) -> bool:
        """Set thinking fields; returns whether thinking is on."""
        mode = self.opts.thinking or "adaptive"
        enabled = thinking.get("enabled")
        effort = thinking.get("effort") or self.opts.effort
        budget = int(thinking.get("budget") or self.opts.thinking_budget or 0)
        always_on = "thinking_always_on" in self.opts.quirks
        if mode == "none":
            return False
        if mode == "budget":
            if enabled is False or (enabled is None and not budget):
                return False
            budget = budget or 4096
            budget = max(1024, min(budget, out["max_tokens"] - 1))
            out["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return True
        if enabled is False and not always_on:
            out["thinking"] = {"type": "disabled"}
            if effort:
                out["output_config"] = {"effort": effort}
            return False
        if enabled is False:  # cannot switch it off: think as little as possible
            effort = "low"
        out["thinking"] = {"type": "adaptive", "display": "summarized"}
        if effort:
            out["output_config"] = {"effort": effort}
        return True

    @staticmethod
    def _add(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
        """Append blocks, merging into the previous message when it has the same role."""
        if messages and messages[-1]["role"] == role:
            prev = messages[-1]["content"]
            if role == "user" and blocks and blocks[0].get("type") == "tool_result":
                # tool results must come before any text in a user message
                n = sum(1 for b in prev if b.get("type") == "tool_result")
                prev[n:n] = blocks
            else:
                prev.extend(blocks)
        else:
            messages.append({"role": role, "content": list(blocks)})

    def _assistant_blocks(self, m: dict[str, Any], strip_thinking: bool) -> list[dict[str, Any]]:
        replay = m.get("replay") or {}
        if replay.get("provider") == self.kind and replay.get("content"):
            blocks = copy.deepcopy(replay["content"])
            if strip_thinking:
                blocks = [b for b in blocks if b.get("type") not in THINKING_BLOCKS]
            if blocks:
                return blocks
        blocks: list[dict[str, Any]] = []
        text = text_of(m.get("content"))
        if text.strip():
            blocks.append({"type": "text", "text": text})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id(tc.get("id", "")),
                    "name": fn.get("name", ""),
                    "input": args if isinstance(args, dict) else {},
                }
            )
        return blocks or [{"type": "text", "text": "(no content)"}]

    # -- chat ----------------------------------------------------------------------------------

    def _chat(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        try:
            return self._send(body, stream, monitors, strip_thinking=False)
        except LLMError as e:
            if not is_signature_error(e):
                raise
            c = self._send(body, stream, monitors, strip_thinking=True)
            c.notes.append(f"thinking blocks stripped and request retried: {e}")
            c.replay_rejected = True
            return c

    def _send(
        self, body: dict[str, Any], stream: bool, monitors: list[Monitor], strip_thinking: bool
    ) -> Completion:
        wire = self.wire_body(body, stream, strip_thinking)
        url = f"{self.base_url}/messages"
        with self.exchange("/messages", body, wire, stream) as ex:
            if stream:
                c = self._stream(url, wire, monitors, ex)
            else:
                t0 = time.perf_counter()
                data = self.post_json(url, wire, ex)
                c = self.parse_message(data)
                c.total_ms = (time.perf_counter() - t0) * 1000
            ex.completion = c
            return c

    def parse_message(self, data: dict[str, Any]) -> Completion:
        c = Completion(raw=data, token_chunks=True)
        blocks = data.get("content") or []
        for b in blocks:
            t = b.get("type")
            if t == "thinking":
                c.reasoning += b.get("thinking") or ""
            elif t == "text":
                c.content += b.get("text") or ""
            elif t == "tool_use":
                c.tool_calls.append(
                    RawToolCall(b.get("id", ""), b.get("name", ""), json.dumps(b.get("input", {})))
                )
        self._finish(c, data.get("stop_reason"), data.get("stop_details"), data.get("model"))
        c.usage = normalise_usage(data.get("usage") or {})
        if any(b.get("type") in THINKING_BLOCKS for b in blocks):
            c.replay = {"provider": self.kind, "model": c.model, "content": blocks}
        return c

    def _finish(
        self, c: Completion, stop: str | None, details: dict[str, Any] | None, model: str | None
    ) -> None:
        c.stop_raw = stop
        c.finish_reason = STOP.get(stop or "", stop)
        if model:
            c.model = model
        if stop == "refusal":
            d = details or {}
            c.refusal = d.get("explanation") or d.get("category") or "refused"
        if stop == "model_context_window_exceeded":
            c.notes.append("generation stopped at the model's context window")

    def _stream(self, url: str, wire: dict[str, Any], monitors: list[Monitor], ex: Any):
        c = Completion(token_chunks=False)
        blocks: dict[int, dict[str, Any]] = {}
        partial: dict[int, str] = {}
        usage: dict[str, Any] = {}
        t0 = time.perf_counter()
        got = [False]
        stop, details = None, None
        with (
            stream_guard(lambda: got[0], self.http.timeout.read),
            self.stream_events(url, wire, ex) as events,
        ):
            try:
                for ev in events:
                    data = ev.json()
                    kind = data.get("type") or ev.event
                    if kind == "error":
                        err = data.get("error") or {}
                        retryable = not got[0] and err.get("type") in (
                            "overloaded_error",
                            "api_error",
                        )
                        raise LLMError(
                            f"stream error: {err.get('type')}: {err.get('message')}",
                            body=data,
                            retryable=retryable,
                        )
                    got[0] = True
                    if kind == "message_start":
                        msg = data.get("message") or {}
                        usage.update(msg.get("usage") or {})
                        c.model = msg.get("model") or c.model
                    elif kind == "content_block_start":
                        i = data.get("index", len(blocks))
                        blocks[i] = dict(data.get("content_block") or {})
                        partial[i] = ""
                    elif kind == "content_block_delta":
                        self._delta(c, blocks, partial, data, t0, monitors)
                    elif kind == "content_block_stop":
                        i = data.get("index")
                        b = blocks.get(i)
                        if b is not None and b.get("type") == "tool_use":
                            b["input"] = _parse_input(partial.get(i, ""))
                    elif kind == "message_delta":
                        d = data.get("delta") or {}
                        stop = d.get("stop_reason") or stop
                        details = d.get("stop_details") or details
                        usage.update({k: v for k, v in (data.get("usage") or {}).items() if v})
                    elif kind == "message_stop":
                        break
            except StopGeneration as s:
                c.aborted = s.reason
                c.abort_detail = s.detail
        c.total_ms = (time.perf_counter() - t0) * 1000
        ordered = [blocks[i] for i in sorted(blocks)]
        for i in sorted(blocks):
            b = blocks[i]
            if b.get("type") == "tool_use":
                args = partial.get(i, "")
                c.tool_calls.append(RawToolCall(b.get("id", ""), b.get("name", ""), args or "{}"))
                if "input" not in b or not isinstance(b["input"], dict):
                    b["input"] = _parse_input(args)
        self._finish(c, stop, details, c.model)
        c.usage = normalise_usage(usage) if usage else {}
        if c.aborted is None and any(b.get("type") in THINKING_BLOCKS for b in ordered):
            c.replay = {"provider": self.kind, "model": c.model, "content": ordered}
        c.raw = {
            "stream": True,
            "content": ordered,
            "stop_reason": stop,
            "usage": usage,
            "aborted": c.aborted,
        }
        return c

    @staticmethod
    def _delta(
        c: Completion,
        blocks: dict[int, dict[str, Any]],
        partial: dict[int, str],
        data: dict[str, Any],
        t0: float,
        monitors: list[Monitor],
    ) -> None:
        i = data.get("index")
        d = data.get("delta") or {}
        b = blocks.setdefault(i, {"type": "text", "text": ""})
        t = d.get("type")
        events: list[tuple[str, str]] = []
        if t == "thinking_delta":
            text = d.get("thinking") or ""
            b["thinking"] = (b.get("thinking") or "") + text
            c.reasoning += text
            c.reasoning_chunks += 1
            if text:
                events.append(("reasoning", text))
        elif t == "signature_delta":
            b["signature"] = (b.get("signature") or "") + (d.get("signature") or "")
        elif t == "text_delta":
            text = d.get("text") or ""
            b["text"] = (b.get("text") or "") + text
            c.content += text
            c.content_chunks += 1
            if text:
                events.append(("content", text))
        elif t == "input_json_delta":
            text = d.get("partial_json") or ""
            partial[i] = partial.get(i, "") + text
            c.content_chunks += 1
            events.append(("tool", text))
        if events:
            mark_ttft(c, t0)
        run_monitors(c, events, monitors)

    # -- other endpoints --------------------------------------------------------------------------

    def count_prompt(self, body: dict[str, Any]) -> int | None:
        wire = self.wire_body({**body, "max_tokens": 1}, stream=False)
        keep = ("model", "system", "messages", "tools", "tool_choice", "thinking")
        payload = {k: v for k, v in wire.items() if k in keep}
        if payload.get("thinking", {}).get("type") == "adaptive":
            payload.pop("thinking")
        if not payload.get("messages"):
            payload["messages"] = [{"role": "user", "content": "x"}]
        data = self.post_json(f"{self.base_url}/messages/count_tokens", payload)
        return int(data["input_tokens"])

    def _detect(self) -> Detected:
        if not self.model:
            return Detected(flavor="anthropic")
        data = self.get_json(f"{self.base_url}/models/{self.model}")
        caps = {"tools", "thinking", "vision"}
        return Detected(
            flavor="anthropic",
            context_window=int(data.get("max_input_tokens") or 0),
            max_output=int(data.get("max_tokens") or 0),
            models=[data.get("id", self.model)],
            capabilities=caps,
            raw={k: data.get(k) for k in ("id", "display_name", "created_at")},
        )

    def tokenize(self, text: str) -> list[int]:
        raise NotSupported("anthropic has no tokenizer endpoint; argus uses count_tokens")

    def health(self) -> bool:
        return bool(self.detect().models)


def _parse_input(text: str) -> Any:
    try:
        value = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def normalise_usage(u: dict[str, Any]) -> dict[str, Any]:
    """Anthropic's usage → the OpenAI-shaped usage argus stores.

    ``input_tokens`` excludes cache reads and writes, so the prompt is their sum.
    """
    read = int(u.get("cache_read_input_tokens") or 0)
    write = int(u.get("cache_creation_input_tokens") or 0)
    prompt = int(u.get("input_tokens") or 0) + read + write
    out: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": int(u.get("output_tokens") or 0),
        "total_tokens": prompt + int(u.get("output_tokens") or 0),
        "prompt_tokens_details": {"cached_tokens": read},
    }
    if write:
        out["cache_write_tokens"] = write
    return out
