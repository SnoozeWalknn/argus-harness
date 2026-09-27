"""Google Gemini API (``models/{model}:generateContent``), streamed over SSE or not.

Translation from the canonical request:

* system messages → ``systemInstruction``; user / assistant → ``contents`` with
  ``user`` / ``model`` roles (consecutive same-role turns merged, as the API wants
  them alternating)
* assistant ``tool_calls`` → ``functionCall`` parts; ``tool`` messages →
  ``functionResponse`` parts (``{"result": text}``), all responses of a turn in one
  user content
* an assistant message with Gemini replay state is sent back as the exact parts
  the model returned, so ``thoughtSignature``\\ s stay attached to their parts;
  for models that require signatures (quirk ``thought_signatures``), function
  calls that came from another model get the documented placeholder signature
* tool schemas are reduced to the subset the API accepts
* thinking → ``thinkingConfig`` (``thinkingBudget``, or ``thinkingLevel`` with
  ``ProviderOptions.thinking = "gemini_level"``) with thought summaries included
* ``model.safety`` → ``safetySettings``: one threshold for each adjustable harm
  category (``off`` turns the filter off). Unset, the API's defaults apply.

Gemini does not always return function-call ids, so argus assigns them.
"""

from __future__ import annotations

import copy
import itertools
import json
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

# For function calls Gemini did not produce itself (e.g. history from another model).
PLACEHOLDER_SIGNATURE = "context_engineering_is_the_way_to_go"
SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "title",
    "propertyOrdering",
}
REFUSALS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
# model.safety → the API's HarmBlockThreshold, applied to each adjustable category
SAFETY_THRESHOLDS = {
    "off": "OFF",
    "block_none": "BLOCK_NONE",
    "block_only_high": "BLOCK_ONLY_HIGH",
    "block_medium_and_above": "BLOCK_MEDIUM_AND_ABOVE",
    "block_low_and_above": "BLOCK_LOW_AND_ABOVE",
}
HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)
EFFORT_BUDGET = {"minimal": 512, "low": 1024, "medium": 8192, "high": 24576}
LOCAL_ID = "gemini_"  # prefix of function-call ids argus assigns when Gemini gives none
_ids = itertools.count(1)


def sanitize_schema(schema: Any) -> Any:
    """JSON schema → the OpenAPI subset ``functionDeclarations`` accept."""
    if isinstance(schema, list):
        return [sanitize_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "const":
            out["enum"] = [value]
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: sanitize_schema(v) for k, v in value.items()}
        elif key in ("items", "anyOf"):
            out[key] = sanitize_schema(value)
        elif key in SCHEMA_KEYS:
            out[key] = value
    return out


class GeminiProvider(Provider):
    kind = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    token_chunks = False
    retry_statuses = (408, 429, 500, 502, 503, 504)

    def auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key} if self.api_key else {}

    @staticmethod
    def strip_replay(replay: dict[str, Any]) -> dict[str, Any] | None:
        parts = []
        for p in replay.get("parts") or []:
            if p.get("thought"):
                continue
            p = {k: v for k, v in p.items() if k != "thoughtSignature"}
            if p and p != {"text": ""}:
                parts.append(p)
        return {**replay, "parts": parts} if parts else None

    # -- request -------------------------------------------------------------------------------

    def wire_body(self, body: dict[str, Any]) -> dict[str, Any]:
        for key in ("grammar", "response_format", "json_schema"):
            if body.get(key):
                raise LLMError(f"gemini does not take {key!r} from argus")
        model = body.get("model") or self.model
        system: list[str] = []
        contents: list[dict[str, Any]] = []
        names: dict[str, str] = {}
        for m in body.get("messages") or []:
            role = m.get("role")
            if role == "system":
                system.append(text_of(m.get("content")))
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    names[tc.get("id", "")] = (tc.get("function") or {}).get("name", "")
                self._add(contents, "model", self._model_parts(m, model))
            elif role == "tool":
                call_id = m.get("tool_call_id", "")
                fr: dict[str, Any] = {
                    "name": names.get(call_id, ""),
                    "response": {"result": text_of(m.get("content"))},
                }
                if call_id and not call_id.startswith(LOCAL_ID):
                    fr["id"] = call_id
                self._add(contents, "user", [{"functionResponse": fr}])
            else:
                self._add(contents, "user", [{"text": text_of(m.get("content")) or " "}])

        out: dict[str, Any] = {"contents": contents}
        if system:
            out["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        tools = body.get("tools") or []
        if tools:
            out["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parameters": sanitize_schema(
                                t["function"].get("parameters") or {"type": "object"}
                            ),
                        }
                        for t in tools
                    ]
                }
            ]
            mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(
                body.get("tool_choice") or "auto", "AUTO"
            )
            out["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        gen: dict[str, Any] = {}
        if body.get("max_tokens"):
            gen["maxOutputTokens"] = int(body["max_tokens"])
        if "no_sampling" not in self.opts.quirks:
            for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK")):
                if body.get(src) is not None:
                    gen[dst] = body[src]
            if body.get("seed") is not None:
                gen["seed"] = body["seed"]
        thinking = self._thinking(body.get("thinking") or {})
        if thinking is not None:
            gen["thinkingConfig"] = thinking
        if gen:
            out["generationConfig"] = gen
        if self.opts.safety:
            threshold = SAFETY_THRESHOLDS[self.opts.safety]
            out["safetySettings"] = [
                {"category": c, "threshold": threshold} for c in HARM_CATEGORIES
            ]
        return out

    def _thinking(self, t: dict[str, Any]) -> dict[str, Any] | None:
        mode = self.opts.thinking or "gemini_budget"
        if mode == "none":
            return None
        enabled = t.get("enabled")
        effort = t.get("effort") or self.opts.effort
        budget = int(t.get("budget") or self.opts.thinking_budget or 0)
        if mode == "gemini_level":
            level = "low" if enabled is False or effort in ("minimal", "low") else "high"
            return {"thinkingLevel": level, "includeThoughts": True}
        if enabled is False:
            return {"thinkingBudget": 0}
        if not budget:
            budget = EFFORT_BUDGET.get(effort, -1)  # -1: the model decides
        return {"thinkingBudget": budget, "includeThoughts": True}

    @staticmethod
    def _add(contents: list[dict[str, Any]], role: str, parts: list[dict[str, Any]]) -> None:
        if contents and contents[-1]["role"] == role:
            prev = contents[-1]["parts"]
            if role == "user" and parts and "functionResponse" in parts[0]:
                n = sum(1 for p in prev if "functionResponse" in p)
                prev[n:n] = parts  # function responses first
            else:
                prev.extend(parts)
        else:
            contents.append({"role": role, "parts": list(parts)})

    def _model_parts(self, m: dict[str, Any], model: str) -> list[dict[str, Any]]:
        replay = m.get("replay") or {}
        if (
            replay.get("provider") == self.kind
            and replay.get("parts")
            and replay.get("model", model) == model
        ):
            return copy.deepcopy(replay["parts"])
        parts: list[dict[str, Any]] = []
        text = text_of(m.get("content"))
        if text.strip():
            parts.append({"text": text})
        signed = False
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            part: dict[str, Any] = {
                "functionCall": {
                    "name": fn.get("name", ""),
                    "args": args if isinstance(args, dict) else {},
                }
            }
            if tc.get("id") and not tc["id"].startswith(LOCAL_ID):
                part["functionCall"]["id"] = tc["id"]
            if "thought_signatures" in self.opts.quirks and not signed:
                part["thoughtSignature"] = PLACEHOLDER_SIGNATURE
                signed = True
            parts.append(part)
        return parts or [{"text": " "}]

    # -- chat ----------------------------------------------------------------------------------

    def _url(self, model: str, method: str) -> str:
        return f"{self.base_url}/models/{model}:{method}"

    def _chat(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        wire = self.wire_body(body)
        model = body.get("model") or self.model
        method = "streamGenerateContent" if stream else "generateContent"
        path = f"/models/{model}:{method}"
        with self.exchange(path, body, wire, stream) as ex:
            if stream:
                c = self._stream(self._url(model, method) + "?alt=sse", wire, monitors, ex)
            else:
                t0 = time.perf_counter()
                data = self.post_json(self._url(model, method), wire, ex)
                c = Completion(raw=data, token_chunks=True)
                parts: list[dict[str, Any]] = []
                self._apply(c, parts, data, None, [])
                self._finish(c, parts, model)
                c.total_ms = (time.perf_counter() - t0) * 1000
            ex.completion = c
            return c

    def _stream(self, url: str, wire: dict[str, Any], monitors: list[Monitor], ex: Any):
        c = Completion(token_chunks=False)
        parts: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        got = [False]
        with (
            stream_guard(lambda: got[0], self.http.timeout.read),
            self.stream_events(url, wire, ex) as events,
        ):
            try:
                for ev in events:
                    data = ev.json()
                    if "error" in data:
                        err = data["error"]
                        msg = f"{err.get('status')}: {err.get('message')}"
                        raise self.error_class(400, msg, data)(f"stream error: {msg}", body=data)
                    got[0] = True
                    self._apply(c, parts, data, t0, monitors)
            except StopGeneration as s:
                c.aborted = s.reason
                c.abort_detail = s.detail
        c.total_ms = (time.perf_counter() - t0) * 1000
        self._finish(c, parts, wire_model(url))
        if c.aborted:
            c.replay = None
        return c

    def _apply(
        self,
        c: Completion,
        parts: list[dict[str, Any]],
        data: dict[str, Any],
        t0: float | None,
        monitors: list[Monitor],
    ) -> None:
        if data.get("usageMetadata"):
            c.usage = normalise_usage(data["usageMetadata"])
        if data.get("modelVersion"):
            c.model = data["modelVersion"]
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            c.refusal = f"prompt blocked: {feedback['blockReason']}{blocked(feedback)}"
            c.stop_raw = feedback["blockReason"]
        for cand in (data.get("candidates") or [])[:1]:
            if cand.get("finishReason"):
                c.stop_raw = cand["finishReason"]
                if cand.get("finishMessage"):
                    c.notes.append(f"{cand['finishReason']}: {cand['finishMessage']}")
                if cand["finishReason"] in REFUSALS and not c.refusal:
                    c.refusal = cand["finishReason"] + blocked(cand)
            events: list[tuple[str, str]] = []
            for part in (cand.get("content") or {}).get("parts") or []:
                _merge_part(parts, part)
                if "functionCall" in part:
                    fc = part["functionCall"]
                    # argus-assigned ids never go back to the API (see wire_body)
                    call_id = fc.get("id") or f"{LOCAL_ID}{next(_ids)}"
                    args = json.dumps(fc.get("args") or {})
                    c.tool_calls.append(RawToolCall(call_id, fc.get("name", ""), args))
                    c.content_chunks += 1
                    events.append(("tool", args))
                elif part.get("thought"):
                    text = part.get("text") or ""
                    c.reasoning += text
                    c.reasoning_chunks += 1
                    if text:
                        events.append(("reasoning", text))
                elif part.get("text"):
                    c.content += part["text"]
                    c.content_chunks += 1
                    events.append(("content", part["text"]))
            if events and t0 is not None:
                mark_ttft(c, t0)
            run_monitors(c, events, monitors)

    def _finish(self, c: Completion, parts: list[dict[str, Any]], model: str) -> None:
        stop = c.stop_raw
        if stop == "MAX_TOKENS":
            c.finish_reason = "length"
        elif stop in REFUSALS or c.refusal:
            c.finish_reason = "refusal"
            c.refusal = c.refusal or stop
            if c.stop_raw == "SAFETY" and self.opts.safety != "off":  # the adjustable filter
                c.notes.append(
                    "Gemini's safety filter blocked this reply; "
                    'model.safety = "off" turns the filter off'
                )
        elif stop == "MALFORMED_FUNCTION_CALL":
            c.finish_reason = "stop"
            c.malformed = "the model produced a malformed function call"
        elif c.tool_calls:
            c.finish_reason = "tool_calls"
        else:
            c.finish_reason = "stop"
        if any("thoughtSignature" in p for p in parts):
            c.replay = {"provider": self.kind, "model": model or self.model, "parts": parts}
        c.raw = {"parts": parts, "finishReason": stop, "usage": c.usage}

    # -- other endpoints --------------------------------------------------------------------------

    def count_prompt(self, body: dict[str, Any]) -> int | None:
        model = body.get("model") or self.model
        wire = self.wire_body({**body, "max_tokens": 0})
        wire.pop("generationConfig", None)
        wire.pop("toolConfig", None)
        wire.pop("safetySettings", None)
        payload = {"generateContentRequest": {"model": f"models/{model}", **wire}}
        data = self.post_json(self._url(model, "countTokens"), payload)
        return int(data.get("totalTokens") or 0)

    def _detect(self) -> Detected:
        if not self.model:
            return Detected(flavor="gemini")
        data = self.get_json(f"{self.base_url}/models/{self.model}")
        caps = {"tools", "vision"} | ({"thinking"} if data.get("thinking") else set())
        return Detected(
            flavor="gemini",
            context_window=int(data.get("inputTokenLimit") or 0),
            max_output=int(data.get("outputTokenLimit") or 0),
            models=[str(data.get("name", self.model)).removeprefix("models/")],
            capabilities=caps,
        )

    def health(self) -> bool:
        return bool(self.detect().models)


def blocked(obj: dict[str, Any]) -> str:
    """The categories a candidate or prompt was blocked for, as ' (dangerous content)'."""
    cats = [
        r.get("category", "").removeprefix("HARM_CATEGORY_").replace("_", " ").lower()
        for r in obj.get("safetyRatings") or []
        if r.get("blocked")
    ]
    return f" ({', '.join(cats)})" if cats else ""


def wire_model(url: str) -> str:
    return url.rsplit("/models/", 1)[-1].split(":", 1)[0]


def _merge_part(parts: list[dict[str, Any]], part: dict[str, Any]) -> None:
    """Streamed text parts are deltas: join them unless a signature or a call separates them."""
    part = copy.deepcopy(part)
    prev = parts[-1] if parts else None
    if (
        prev is not None
        and "text" in part
        and "text" in prev
        and "thoughtSignature" not in prev
        and "functionCall" not in prev
        and bool(prev.get("thought")) == bool(part.get("thought"))
    ):
        prev["text"] += part["text"]
        if "thoughtSignature" in part:
            prev["thoughtSignature"] = part["thoughtSignature"]
        return
    parts.append(part)


def normalise_usage(u: dict[str, Any]) -> dict[str, Any]:
    prompt = int(u.get("promptTokenCount") or 0)
    thoughts = int(u.get("thoughtsTokenCount") or 0)
    out_tokens = int(u.get("candidatesTokenCount") or 0) + thoughts
    return {
        "prompt_tokens": prompt,
        "completion_tokens": out_tokens,
        "total_tokens": int(u.get("totalTokenCount") or prompt + out_tokens),
        "prompt_tokens_details": {"cached_tokens": int(u.get("cachedContentTokenCount") or 0)},
        "completion_tokens_details": {"reasoning_tokens": thoughts},
    }
