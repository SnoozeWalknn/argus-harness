"""The Gemini API format for the mock server.

Converts ``models/{model}:generateContent`` requests to the canonical chat shape,
checks them the way the API does (alternating roles, one function response per
function call, the schema subset ``functionDeclarations`` accept) and renders
scripted steps as ``GenerateContentResponse`` objects or SSE chunks.

For ``gemini-3*`` models the mock also enforces thought signatures: the first
function call of every model turn must carry a signature the mock issued (or the
documented placeholder), as those models require.

A step with ``"harm": "dangerous_content"`` (a harm category, lower case without
the ``HARM_CATEGORY_`` prefix) stands for a reply the safety filter rates HIGH in
that category: it is blocked with ``finishReason: SAFETY`` unless the request's
``safetySettings`` set that category to ``BLOCK_NONE`` or ``OFF``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PLACEHOLDER = "context_engineering_is_the_way_to_go"
HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)
THRESHOLDS = (
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
    "OFF",
)
UNSUPPORTED_SCHEMA = ("additionalProperties", "$schema", "const", "$ref", "patternProperties")


def requires_signatures(model: str) -> bool:
    return str(model).startswith("gemini-3")


def sign(model: str, text: str) -> str:
    return f"mockgsig.{model}.{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def to_canonical(model: str, req: dict[str, Any]) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    sys_parts = (req.get("systemInstruction") or {}).get("parts") or []
    system = "".join(p.get("text", "") for p in sys_parts)
    if system:
        msgs.append({"role": "system", "content": system})
    for content in req.get("contents") or []:
        parts = content.get("parts") or []
        if content.get("role") == "model":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(
                    p.get("text", "") for p in parts if "text" in p and not p.get("thought")
                ),
            }
            calls = [
                {
                    "id": p["functionCall"].get("id") or p["functionCall"].get("name"),
                    "type": "function",
                    "function": {
                        "name": p["functionCall"].get("name"),
                        "arguments": json.dumps(p["functionCall"].get("args") or {}),
                    },
                }
                for p in parts
                if "functionCall" in p
            ]
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
            continue
        texts = []
        for p in parts:
            if "functionResponse" in p:
                fr = p["functionResponse"]
                resp = fr.get("response") or {}
                out = resp.get("result", resp.get("output", resp))
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": fr.get("id") or fr.get("name"),
                        "content": out if isinstance(out, str) else json.dumps(out),
                    }
                )
            elif "text" in p:
                texts.append(p["text"])
        if texts:
            msgs.append({"role": "user", "content": "\n".join(texts)})
    gen = req.get("generationConfig") or {}
    canon: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": gen.get("maxOutputTokens"),
    }
    decls = [d for t in req.get("tools") or [] for d in t.get("functionDeclarations") or []]
    if decls:
        canon["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": d.get("name"),
                    "description": d.get("description", ""),
                    "parameters": d.get("parameters") or {},
                },
            }
            for d in decls
        ]
    return canon


def _schema_problem(schema: Any, where: str) -> str | None:
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in UNSUPPORTED_SCHEMA:
                return f"Invalid JSON payload received. Unknown name \"{key}\" at '{where}'"
            if key == "properties" and isinstance(value, dict):
                for name, sub in value.items():
                    p = _schema_problem(sub, f"{where}.properties[{name}]")
                    if p:
                        return p
            elif key in ("items", "anyOf"):
                p = _schema_problem(value, f"{where}.{key}")
                if p:
                    return p
    elif isinstance(schema, list):
        for i, sub in enumerate(schema):
            p = _schema_problem(sub, f"{where}[{i}]")
            if p:
                return p
    return None


def validate(model: str, req: dict[str, Any], verify: bool = True) -> str | None:
    contents = req.get("contents") or []
    if not contents:
        return "* GenerateContentRequest.contents: contents is not specified"
    seen: set[str] = set()
    for i, s in enumerate(req.get("safetySettings") or []):
        if s.get("category") not in HARM_CATEGORIES:
            return (
                f"Invalid value at 'safety_settings[{i}].category' "
                f"(type.googleapis.com/google.ai.generativelanguage.v1beta.HarmCategory), "
                f'"{s.get("category")}"'
            )
        if s.get("threshold") not in THRESHOLDS:
            return (
                f"Invalid value at 'safety_settings[{i}].threshold' "
                f"(type.googleapis.com/google.ai.generativelanguage.v1beta."
                f'SafetySetting.HarmBlockThreshold), "{s.get("threshold")}"'
            )
        if s["category"] in seen:
            return f"* GenerateContentRequest.safety_settings: duplicate category {s['category']}"
        seen.add(s["category"])
    for t in req.get("tools") or []:
        for i, d in enumerate(t.get("functionDeclarations") or []):
            p = _schema_problem(
                d.get("parameters"), f"tools[0].function_declarations[{i}].parameters"
            )
            if p:
                return p
    prev_role = None
    for i, content in enumerate(contents):
        role = content.get("role")
        if role not in ("user", "model"):
            return f"Please use a valid role: user, model. (contents[{i}])"
        if role == prev_role:
            return "Please ensure that multiturn requests alternate between user and model."
        prev_role = role
        parts = content.get("parts") or []
        if not parts:
            return f"contents[{i}].parts: must not be empty"
        if role == "model":
            n_calls = sum(1 for p in parts if "functionCall" in p)
            nxt = contents[i + 1] if i + 1 < len(contents) else None
            n_resp = sum(1 for p in (nxt or {}).get("parts") or [] if "functionResponse" in p)
            if n_calls and nxt is not None and n_calls != n_resp:
                return (
                    "Please ensure that the number of function response parts is equal to the "
                    "number of function call parts of the function call turn."
                )
            first = next((p for p in parts if "functionCall" in p), None)
            if verify and requires_signatures(model) and first is not None:
                sig = str(first.get("thoughtSignature") or "")
                if sig != PLACEHOLDER and not sig.startswith(f"mockgsig.{model}."):
                    name = first["functionCall"].get("name")
                    return (
                        f"Function call is missing a thought_signature in functionCall parts. "
                        f"This is required for tools to work correctly ({name}, contents[{i}])."
                    )
    return None


def error_body(code: int, status: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "status": status}}


def harm_blocked(step: dict[str, Any], req: dict[str, Any]) -> str | None:
    """The category a scripted ``harm`` step is blocked for under the request's settings."""
    if not step.get("harm"):
        return None
    category = "HARM_CATEGORY_" + str(step["harm"]).upper()
    for s in req.get("safetySettings") or []:
        if s.get("category") == category and s.get("threshold") in ("BLOCK_NONE", "OFF"):
            return None
    return category  # rated HIGH: every other threshold (and the default) blocks it


def finish_reason(step: dict[str, Any], finish: str) -> str:
    if step.get("refusal"):
        return "SAFETY"
    if step.get("stop_reason"):
        return step["stop_reason"]
    return {"length": "MAX_TOKENS"}.get(finish, "STOP")


def usage(n_prompt: int, cached: int, n_content: int, n_thoughts: int) -> dict[str, Any]:
    out = {
        "promptTokenCount": n_prompt,
        "candidatesTokenCount": n_content,
        "totalTokenCount": n_prompt + n_content + n_thoughts,
    }
    if n_thoughts:
        out["thoughtsTokenCount"] = n_thoughts
    if cached:
        out["cachedContentTokenCount"] = cached
    return out


def chunks(
    model: str,
    step: dict[str, Any],
    pieces: list[tuple[str, Any]],
    calls: list[dict[str, Any]],
    finish: str,
    usage_meta: dict[str, Any],
    show_thoughts: bool,
    with_ids: bool,
) -> list[dict[str, Any]]:
    """Streamed chunks; the non-streamed response is these merged."""
    thought = "".join(p for k, p in pieces if k == "reasoning")
    signature = sign(model, thought)
    out: list[dict[str, Any]] = []

    def cand(parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0}],
            "modelVersion": model,
            "responseId": "mock-resp",
        }

    for kind, piece in pieces:
        if kind == "reasoning" and show_thoughts:
            out.append(cand([{"text": piece, "thought": True}]))
        elif kind == "content":
            out.append(cand([{"text": piece}]))
    fc_parts = []
    for i, tc in enumerate(calls):
        try:
            args = json.loads(tc["arguments"])
        except json.JSONDecodeError:
            continue  # a call cut off by the limit is not emitted
        fc: dict[str, Any] = {"name": tc["name"], "args": args}
        if with_ids:
            fc["id"] = tc["id"]
        part: dict[str, Any] = {"functionCall": fc}
        if i == 0:
            part["thoughtSignature"] = signature
        fc_parts.append(part)
    if fc_parts:
        out.append(cand(fc_parts))
    elif thought:
        out.append(cand([{"text": "", "thoughtSignature": signature}]))
    last = cand([]) if not out else out.pop()
    last["candidates"][0]["finishReason"] = finish
    if step.get("harm"):
        category = "HARM_CATEGORY_" + str(step["harm"]).upper()
        rating = {"category": category, "probability": "HIGH"}
        if finish == "SAFETY":
            rating["blocked"] = True
        last["candidates"][0]["safetyRatings"] = [rating]
    if not last["candidates"][0]["content"]["parts"]:
        del last["candidates"][0]["content"]
    last["usageMetadata"] = usage_meta
    out.append(last)
    return out


def merged(chunks_: list[dict[str, Any]]) -> dict[str, Any]:
    """Non-streamed response: the chunks' parts joined the way the API returns them."""
    parts: list[dict[str, Any]] = []
    for ch in chunks_:
        for p in ((ch["candidates"][0].get("content") or {}).get("parts")) or []:
            prev = parts[-1] if parts else None
            if (
                prev
                and "text" in p
                and "text" in prev
                and "thoughtSignature" not in prev
                and bool(prev.get("thought")) == bool(p.get("thought"))
            ):
                prev["text"] += p["text"]
                if "thoughtSignature" in p:
                    prev["thoughtSignature"] = p["thoughtSignature"]
            else:
                parts.append(dict(p))
    last = chunks_[-1]
    candidate = {
        "content": {"role": "model", "parts": parts},
        "finishReason": last["candidates"][0].get("finishReason"),
        "index": 0,
    }
    if last["candidates"][0].get("safetyRatings"):
        candidate["safetyRatings"] = last["candidates"][0]["safetyRatings"]
    return {
        "candidates": [candidate],
        "usageMetadata": last.get("usageMetadata"),
        "modelVersion": last.get("modelVersion"),
        "responseId": last.get("responseId"),
    }
