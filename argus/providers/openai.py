"""OpenAI Responses API (``POST /v1/responses``), streamed or not.

Stateless use (``store: false``): the whole conversation is sent every turn as
input items, and reasoning is carried across tool calls by replaying the
``reasoning`` items with their ``encrypted_content`` (requested with
``include: ["reasoning.encrypted_content"]``). Encrypted reasoning only
verifies for the model that produced it, so it is replayed to that model only.

Translation from the canonical request:

* system messages → ``instructions``; user messages → ``{"role": "user", ...}``
* assistant ``tool_calls`` → ``function_call`` items; ``tool`` messages →
  ``function_call_output`` items; an assistant message with OpenAI replay state is
  sent as the output items the API returned (ids removed except on reasoning items)
* ``max_tokens`` → ``max_output_tokens``; thinking → ``reasoning.effort`` with
  summaries requested so reasoning can be logged; sampling is dropped for
  reasoning models (quirk ``no_sampling``) and ``top_k``-style keys always

Chat Completions against OpenAI still works through ``openai_compat`` with the
``max_completion_tokens`` quirk.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from argus.providers.base import (
    Completion,
    Detected,
    LLMError,
    Monitor,
    Provider,
    RawToolCall,
    StopGeneration,
    mark_ttft,
    run_monitors,
    stream_guard,
    text_of,
)

SAMPLING = ("temperature", "top_p")
DROP_ITEM_KEYS = ("id", "status")


class OpenAIProvider(Provider):
    kind = "openai"
    default_base_url = "https://api.openai.com/v1"
    token_chunks = False
    retry_statuses = (408, 429, 500, 502, 503, 504)

    @staticmethod
    def strip_replay(replay: dict[str, Any]) -> dict[str, Any] | None:
        items = [i for i in replay.get("items") or [] if i.get("type") != "reasoning"]
        return {**replay, "items": items} if items else None

    # -- request -------------------------------------------------------------------------------

    def wire_body(self, body: dict[str, Any], stream: bool) -> dict[str, Any]:
        for key in ("grammar", "response_format", "json_schema"):
            if body.get(key):
                raise LLMError(f"openai (Responses API) does not take {key!r} from argus")
        q = self.opts.quirks
        model = body.get("model") or self.model
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role == "system":
                instructions.append(text_of(m.get("content")))
            elif role == "assistant":
                items += self._assistant_items(m, model)
            elif role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.get("tool_call_id", ""),
                        "output": text_of(m.get("content")),
                    }
                )
            else:
                items.append({"role": "user", "content": text_of(m.get("content"))})

        out: dict[str, Any] = {"model": model}
        if instructions:
            out["instructions"] = "\n\n".join(instructions)
        out["input"] = items
        if body.get("max_tokens"):
            out["max_output_tokens"] = int(body["max_tokens"])
        tools = body.get("tools") or []
        if tools:
            out["tools"] = [
                {
                    "type": "function",
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters") or {"type": "object"},
                    "strict": False,
                }
                for t in tools
            ]
            out["tool_choice"] = body.get("tool_choice") or "auto"
            if "parallel_tool_calls" in body and "no_parallel_tools" not in q:
                out["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
        reasoning = self._reasoning(body.get("thinking") or {})
        if reasoning is not None:
            out["reasoning"] = reasoning
            out["include"] = ["reasoning.encrypted_content"]
        out["store"] = False
        if "no_sampling" not in q:
            for key in SAMPLING:
                if body.get(key) is not None:
                    out[key] = body[key]
        if stream:
            out["stream"] = True
        return out

    def _reasoning(self, thinking: dict[str, Any]) -> dict[str, Any] | None:
        if self.opts.thinking == "none":
            return None
        enabled = thinking.get("enabled")
        effort = thinking.get("effort") or self.opts.effort
        out: dict[str, Any] = {}
        if enabled is False:
            out["effort"] = "minimal"
        elif effort:
            out["effort"] = effort
        if enabled is not False and "no_reasoning_summary" not in self.opts.quirks:
            out["summary"] = "auto"
        return out

    def _assistant_items(self, m: dict[str, Any], model: str) -> list[dict[str, Any]]:
        replay = m.get("replay") or {}
        if replay.get("provider") == self.kind and replay.get("items"):
            same_model = replay.get("model") == model
            out = []
            for item in copy.deepcopy(replay["items"]):
                if item.get("type") == "reasoning":
                    if same_model and item.get("encrypted_content"):
                        item.pop("status", None)
                        out.append(item)
                    continue
                for key in DROP_ITEM_KEYS:
                    item.pop(key, None)
                out.append(item)
            if out:
                return out
        items: list[dict[str, Any]] = []
        text = text_of(m.get("content"))
        if text.strip():
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                }
            )
        return items

    # -- chat ----------------------------------------------------------------------------------

    def _chat(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        wire = self.wire_body(body, stream)
        url = f"{self.base_url}/responses"
        with self.exchange("/responses", body, wire, stream) as ex:
            if stream:
                c = self._stream(url, wire, monitors, ex)
            else:
                t0 = time.perf_counter()
                data = self.post_json(url, wire, ex)
                c = self.parse_response(data)
                c.total_ms = (time.perf_counter() - t0) * 1000
            ex.completion = c
            return c

    def parse_response(self, data: dict[str, Any], c: Completion | None = None) -> Completion:
        """Fill a completion from a response object (non-streamed, or streamed + final)."""
        streamed = c is not None
        c = c or Completion(raw=data, token_chunks=True)
        output = data.get("output") or []
        if not streamed:
            summaries = []
            for item in output:
                t = item.get("type")
                if t == "reasoning":
                    summaries += [s.get("text", "") for s in item.get("summary") or []]
                elif t == "message":
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text":
                            c.content += part.get("text", "")
                        elif part.get("type") == "refusal":
                            c.refusal = (c.refusal or "") + part.get("refusal", "")
                elif t == "function_call":
                    c.tool_calls.append(
                        RawToolCall(
                            item.get("call_id") or item.get("id", ""),
                            item.get("name", ""),
                            item.get("arguments") or "{}",
                        )
                    )
            c.reasoning = "\n\n".join(s for s in summaries if s)
        c.model = data.get("model") or c.model
        status = data.get("status")
        c.stop_raw = status
        reason = (data.get("incomplete_details") or {}).get("reason")
        if status == "incomplete" and reason == "max_output_tokens":
            c.finish_reason = "length"
            c.stop_raw = reason
        elif (status == "incomplete" and reason == "content_filter") or c.refusal:
            c.finish_reason = "refusal"
            c.refusal = c.refusal or "content_filter"
        elif c.tool_calls:
            c.finish_reason = "tool_calls"
        else:
            c.finish_reason = "stop"
        c.usage = normalise_usage(data.get("usage") or {})
        if any(i.get("type") == "reasoning" for i in output):
            c.replay = {"provider": self.kind, "model": c.model or self.model, "items": output}
        return c

    def _stream(self, url: str, wire: dict[str, Any], monitors: list[Monitor], ex: Any):
        c = Completion(token_chunks=False)
        items: dict[int, dict[str, Any]] = {}
        args: dict[int, str] = {}
        final: dict[str, Any] | None = None
        t0 = time.perf_counter()
        got = [False]
        last_summary: tuple[int, int] | None = None
        with (
            stream_guard(lambda: got[0], self.http.timeout.read),
            self.stream_events(url, wire, ex) as events,
        ):
            try:
                for ev in events:
                    data = ev.json()
                    kind = data.get("type") or ev.event
                    if kind in ("error", "response.failed"):
                        err = data.get("error") or (data.get("response") or {}).get("error") or {}
                        msg = f"{err.get('code') or kind}: {err.get('message') or data}"
                        cls = self.error_class(400, msg, data)
                        raise cls(f"stream error: {msg}", body=data)
                    got[0] = True
                    i = data.get("output_index", 0)
                    events_out: list[tuple[str, str]] = []
                    if kind in ("response.output_item.added", "response.output_item.done"):
                        items[i] = dict(data.get("item") or {})
                    elif kind == "response.reasoning_summary_text.delta":
                        key = (i, data.get("summary_index", 0))
                        if last_summary is not None and key != last_summary and c.reasoning:
                            c.reasoning += "\n\n"
                        last_summary = key
                        c.reasoning += data.get("delta", "")
                        c.reasoning_chunks += 1
                        events_out.append(("reasoning", data.get("delta", "")))
                    elif kind == "response.output_text.delta":
                        c.content += data.get("delta", "")
                        c.content_chunks += 1
                        events_out.append(("content", data.get("delta", "")))
                    elif kind == "response.refusal.delta":
                        c.refusal = (c.refusal or "") + data.get("delta", "")
                    elif kind == "response.function_call_arguments.delta":
                        args[i] = args.get(i, "") + data.get("delta", "")
                        c.content_chunks += 1
                        events_out.append(("tool", data.get("delta", "")))
                    elif kind in ("response.completed", "response.incomplete"):
                        final = data.get("response") or {}
                        break
                    if events_out:
                        mark_ttft(c, t0)
                        run_monitors(c, events_out, monitors)
            except StopGeneration as s:
                c.aborted = s.reason
                c.abort_detail = s.detail
        c.total_ms = (time.perf_counter() - t0) * 1000
        ordered = [items[i] for i in sorted(items)]
        for i in sorted(items):
            item = items[i]
            if item.get("type") == "function_call":
                arguments = item.get("arguments") or args.get(i, "") or "{}"
                if c.aborted:
                    arguments = args.get(i, "")
                c.tool_calls.append(
                    RawToolCall(
                        item.get("call_id") or item.get("id", ""), item.get("name", ""), arguments
                    )
                )
        if final is not None:
            final = {**final, "output": final.get("output") or ordered}
            self.parse_response(final, c)
        else:
            c.finish_reason = "tool_calls" if c.tool_calls else "stop"
        if c.aborted:
            c.replay = None
        c.raw = {"stream": True, "output": ordered, "status": (final or {}).get("status")}
        return c

    def _detect(self) -> Detected:
        return Detected(flavor="openai", models=[self.model] if self.model else [])

    def health(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/models").status_code == 200
        except Exception:
            return False


def normalise_usage(u: dict[str, Any]) -> dict[str, Any]:
    if not u:
        return {}
    out: dict[str, Any] = {
        "prompt_tokens": int(u.get("input_tokens") or 0),
        "completion_tokens": int(u.get("output_tokens") or 0),
        "total_tokens": int(u.get("total_tokens") or 0),
        "prompt_tokens_details": {
            "cached_tokens": int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        },
    }
    reasoning = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is not None:
        out["completion_tokens_details"] = {"reasoning_tokens": int(reasoning)}
    return out
