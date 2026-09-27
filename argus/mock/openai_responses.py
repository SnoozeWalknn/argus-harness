"""The OpenAI Responses API format for the mock server.

Converts ``/v1/responses`` requests to the canonical chat shape, checks them the
way the API does (unsupported parameters, unmatched function-call outputs,
reasoning items that cannot be used with ``store: false`` or with another
model), and renders scripted steps as response objects or SSE event streams.
Encrypted reasoning is bound to the model that produced it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

KNOWN_KEYS = {
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "reasoning",
    "include",
    "store",
    "stream",
    "temperature",
    "top_p",
    "text",
    "metadata",
    "prompt_cache_key",
    "service_tier",
    "truncation",
    "user",
}


def is_reasoning_model(model: str) -> bool:
    return str(model).startswith(("gpt-5", "o1", "o3", "o4"))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "")
        for p in content or []
        if p.get("type") in ("input_text", "output_text", "text")
    )


def to_canonical(req: dict[str, Any]) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    if req.get("instructions"):
        msgs.append({"role": "system", "content": req["instructions"]})
    current: dict[str, Any] | None = None  # the assistant turn being assembled
    for item in req.get("input") or []:
        t = item.get("type", "message")
        if t == "message" and item.get("role") in ("user", "system", "developer"):
            role = "system" if item["role"] in ("system", "developer") else "user"
            msgs.append({"role": role, "content": _content_text(item.get("content"))})
            current = None
        elif t == "message":
            current = {"role": "assistant", "content": _content_text(item.get("content"))}
            msgs.append(current)
        elif t == "function_call":
            if current is None:
                current = {"role": "assistant", "content": ""}
                msgs.append(current)
            current.setdefault("tool_calls", []).append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
        elif t == "function_call_output":
            msgs.append(
                {"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")}
            )
            current = None
        elif t == "reasoning":
            if current is None:
                current = {"role": "assistant", "content": ""}
                msgs.append(current)
    canon: dict[str, Any] = {
        "model": req.get("model"),
        "messages": msgs,
        "max_tokens": req.get("max_output_tokens"),
    }
    if req.get("tools"):
        canon["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {},
                },
            }
            for t in req["tools"]
        ]
    return canon


def encrypt(model: str, text: str) -> str:
    return f"mockenc.{model}.{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def validate(req: dict[str, Any], verify: bool = True) -> tuple[int, str, str] | None:
    """(status, code, message) for the first thing the API would reject."""
    for key in req:
        if key not in KNOWN_KEYS:
            return 400, "unknown_parameter", f"Unsupported parameter: '{key}'."
    model = str(req.get("model", ""))
    if is_reasoning_model(model):
        for key in ("temperature", "top_p"):
            if key in req:
                return (
                    400,
                    "unsupported_parameter",
                    f"Unsupported parameter: '{key}' is not supported with this model.",
                )
    calls: set[str] = set()
    outputs: set[str] = set()
    for item in req.get("input") or []:
        t = item.get("type", "message")
        if t == "function_call":
            calls.add(item.get("call_id"))
        elif t == "function_call_output":
            if item.get("call_id") not in calls:
                return (
                    400,
                    "invalid_request_error",
                    f"No tool call found for function call output with call_id {item.get('call_id')}.",
                )
            outputs.add(item.get("call_id"))
        elif t == "reasoning":
            enc = item.get("encrypted_content")
            if req.get("store") is False and not enc:
                return (
                    404,
                    "invalid_request_error",
                    f"Item with id '{item.get('id')}' not found. Items are not persisted when "
                    "`store` is set to false.",
                )
            if verify and enc and not str(enc).startswith(f"mockenc.{model}."):
                return (
                    400,
                    "invalid_encrypted_content",
                    f"The encrypted content for item {item.get('id')} could not be verified.",
                )
    missing = calls - outputs
    if missing:
        return (
            400,
            "invalid_request_error",
            f"No tool output found for function call {min(missing)}.",
        )
    return None


def error_body(code: str, message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def build_output(
    req: dict[str, Any], step: dict[str, Any], g: Any, pieces: list[tuple[str, Any]], counter: Any
) -> list[dict[str, Any]]:
    """Output items for a generation: reasoning, message, function calls (in order)."""
    model = str(req.get("model", ""))
    reasoning_cfg = req.get("reasoning") or {}
    items: list[dict[str, Any]] = []
    thought = "".join(p for k, p in pieces if k == "reasoning")
    if thought or (is_reasoning_model(model) and reasoning_cfg):
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": f"rs_mock_{next(counter)}",
            "summary": [],
        }
        if thought and reasoning_cfg.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": thought}]
        if "reasoning.encrypted_content" in (req.get("include") or []):
            item["encrypted_content"] = encrypt(model, thought)
        items.append(item)
    text = "".join(p for k, p in pieces if k == "content")
    if step.get("refusal"):
        refusal = step["refusal"] if isinstance(step["refusal"], str) else "I can't help with that."
        items.append(
            {
                "type": "message",
                "id": f"msg_mock_{next(counter)}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": refusal}],
            }
        )
    elif text:
        items.append(
            {
                "type": "message",
                "id": f"msg_mock_{next(counter)}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    args: dict[str, str] = {}
    for kind, piece in pieces:
        if kind == "tool":
            args[piece[0]["id"]] = args.get(piece[0]["id"], "") + piece[1]
    for tc in g.tool_calls:
        items.append(
            {
                "type": "function_call",
                "id": f"fc_mock_{next(counter)}",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": args.get(tc["id"], ""),
                "status": "completed",
            }
        )
    return items


def response_object(
    rid: str,
    req: dict[str, Any],
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "id": rid,
        "object": "response",
        "created_at": 1758902400,
        "status": "incomplete" if truncated else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
        "model": req.get("model"),
        "output": output,
        "parallel_tool_calls": req.get("parallel_tool_calls", True),
        "store": req.get("store", True),
        "usage": usage,
        "error": None,
    }


def usage(n_prompt: int, cached: int, n_out: int, n_reasoning: int) -> dict[str, Any]:
    return {
        "input_tokens": n_prompt,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": n_out,
        "output_tokens_details": {"reasoning_tokens": n_reasoning},
        "total_tokens": n_prompt + n_out,
    }


def stream_events(
    base: dict[str, Any], output: list[dict[str, Any]], pieces: list[tuple[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """The SSE events for a response, deltas split like the pieces were generated."""
    evs: list[tuple[str, dict[str, Any]]] = []
    start = {**base, "status": "in_progress", "output": [], "usage": None}
    evs.append(("response.created", {"response": start}))
    evs.append(("response.in_progress", {"response": start}))
    by_kind: dict[str, list[str]] = {"reasoning": [], "content": []}
    call_pieces: dict[str, list[str]] = {}
    for kind, piece in pieces:
        if kind == "tool":
            call_pieces.setdefault(piece[0]["id"], []).append(piece[1])
        else:
            by_kind[kind].append(piece)
    for i, item in enumerate(output):
        t = item["type"]
        head = {**item}
        if t == "reasoning":
            head = {"type": "reasoning", "id": item["id"], "summary": []}
        elif t == "message":
            head = {**item, "status": "in_progress", "content": []}
        elif t == "function_call":
            head = {**item, "status": "in_progress", "arguments": ""}
        evs.append(("response.output_item.added", {"output_index": i, "item": head}))
        if t == "reasoning" and item["summary"]:
            ref = {"item_id": item["id"], "output_index": i, "summary_index": 0}
            evs.append(
                (
                    "response.reasoning_summary_part.added",
                    {**ref, "part": {"type": "summary_text", "text": ""}},
                )
            )
            for piece in by_kind["reasoning"]:
                evs.append(("response.reasoning_summary_text.delta", {**ref, "delta": piece}))
            evs.append(
                (
                    "response.reasoning_summary_text.done",
                    {**ref, "text": item["summary"][0]["text"]},
                )
            )
        elif t == "message":
            ref = {"item_id": item["id"], "output_index": i, "content_index": 0}
            part = item["content"][0]
            if part["type"] == "refusal":
                evs.append(
                    (
                        "response.content_part.added",
                        {**ref, "part": {"type": "refusal", "refusal": ""}},
                    )
                )
                evs.append(("response.refusal.delta", {**ref, "delta": part["refusal"]}))
                evs.append(("response.refusal.done", {**ref, "refusal": part["refusal"]}))
            else:
                evs.append(
                    (
                        "response.content_part.added",
                        {**ref, "part": {"type": "output_text", "text": "", "annotations": []}},
                    )
                )
                for piece in by_kind["content"]:
                    evs.append(("response.output_text.delta", {**ref, "delta": piece}))
                evs.append(("response.output_text.done", {**ref, "text": part["text"]}))
            evs.append(("response.content_part.done", {**ref, "part": part}))
        elif t == "function_call":
            ref = {"item_id": item["id"], "output_index": i}
            for piece in call_pieces.get(item["call_id"], []):
                evs.append(("response.function_call_arguments.delta", {**ref, "delta": piece}))
            evs.append(
                ("response.function_call_arguments.done", {**ref, "arguments": item["arguments"]})
            )
        evs.append(("response.output_item.done", {"output_index": i, "item": item}))
    final = "response.incomplete" if base["status"] == "incomplete" else "response.completed"
    evs.append((final, {"response": base}))
    return [
        (name, {"type": name, "sequence_number": n, **data}) for n, (name, data) in enumerate(evs)
    ]


def dumps(value: Any) -> str:
    return json.dumps(value)
