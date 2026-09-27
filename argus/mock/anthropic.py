"""The Anthropic Messages API format for the mock server.

The mock converts an incoming ``/v1/messages`` request to the canonical
OpenAI-shaped request (so scripts, expectations and token counting work as for
chat completions), checks it the way the API does, and renders scripted steps
as Messages API responses or SSE streams.

Thinking blocks get a signature that binds them to the conversation prefix
that produced them (system, tools and earlier messages, ignoring
``cache_control``) and to their text. Replaying a block after the prefix
changed yields the API's "bound to a different conversation" error; editing
the block's text yields the plain invalid-signature error.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
STOP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


def _blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def system_text(req: dict[str, Any]) -> str:
    return _text(req.get("system") or "")


def to_canonical(req: dict[str, Any]) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    system = system_text(req)
    if system:
        msgs.append({"role": "system", "content": system})
    for m in req.get("messages") or []:
        blocks = _blocks(m.get("content"))
        if m.get("role") == "user":
            texts = []
            for b in blocks:
                if b.get("type") == "tool_result":
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id"),
                            "content": _text(b.get("content")),
                        }
                    )
                elif b.get("type") == "text":
                    texts.append(b.get("text", ""))
            if texts:
                msgs.append({"role": "user", "content": "\n".join(texts)})
        else:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
            }
            calls = [
                {
                    "id": b.get("id"),
                    "type": "function",
                    "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input"))},
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
    canon: dict[str, Any] = {
        "model": req.get("model"),
        "messages": msgs,
        "max_tokens": req.get("max_tokens"),
    }
    if req.get("tools"):
        canon["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {},
                },
            }
            for t in req["tools"]
        ]
    return canon


def validate(req: dict[str, Any]) -> list[str]:
    """What the real API would reject with a 400 (the parts argus could get wrong)."""
    problems: list[str] = []
    if not isinstance(req.get("max_tokens"), int):
        problems.append("max_tokens: Field required")
    msgs = req.get("messages") or []
    if not msgs:
        problems.append("messages: at least one message is required")
    elif msgs[0].get("role") != "user":
        problems.append("messages: the first message must use the user role")
    for t in req.get("tools") or []:
        if not TOOL_NAME.match(str(t.get("name", ""))):
            problems.append(f"tools: invalid tool name {t.get('name')!r}")
        if "input_schema" not in t:
            problems.append(f"tools.{t.get('name')}.input_schema: Field required")
    for i, m in enumerate(msgs):
        blocks = _blocks(m.get("content"))
        if m.get("role") not in ("user", "assistant"):
            problems.append(f"messages.{i}.role: unexpected role {m.get('role')!r}")
        for j, b in enumerate(blocks):
            if b.get("type") == "text" and not str(b.get("text", "")).strip():
                problems.append(f"messages.{i}.content.{j}: text content blocks must be non-empty")
        if m.get("role") == "user":
            seen_text = False
            prev = msgs[i - 1] if i else {}
            prev_ids = {
                b.get("id") for b in _blocks(prev.get("content")) if b.get("type") == "tool_use"
            }
            for b in blocks:
                if b.get("type") == "text":
                    seen_text = True
                elif b.get("type") == "tool_result":
                    if seen_text:
                        problems.append(
                            f"messages.{i}: tool_result blocks must come before any text"
                        )
                    if b.get("tool_use_id") not in prev_ids:
                        problems.append(
                            f"messages.{i}: unexpected tool_use_id found in tool_result blocks: "
                            f"{b.get('tool_use_id')}"
                        )
        else:
            ids = [b.get("id") for b in blocks if b.get("type") == "tool_use"]
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            results = {
                b.get("tool_use_id")
                for b in _blocks((nxt or {}).get("content"))
                if b.get("type") == "tool_result"
            }
            missing = [x for x in ids if x not in results]
            if ids and missing:
                problems.append(
                    f"messages.{i}: tool_use ids were found without tool_result blocks "
                    f"immediately after: {', '.join(map(str, missing))}"
                )
    thinking = req.get("thinking") or {}
    if thinking.get("type") in ("enabled", "adaptive"):
        if req.get("temperature") not in (None, 1, 1.0):
            problems.append("temperature may only be set to 1 when thinking is enabled")
        if req.get("top_k") is not None:
            problems.append("top_k is not supported when thinking is enabled")
    if thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens") or 0
        if budget < 1024:
            problems.append("thinking.budget_tokens: must be at least 1024")
        if isinstance(req.get("max_tokens"), int) and budget >= req["max_tokens"]:
            problems.append("max_tokens must be greater than thinking.budget_tokens")
    return problems


def _strip_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache(v) for v in value]
    return value


def prefix_hash(req: dict[str, Any], upto: int) -> str:
    blob = json.dumps(
        _strip_cache([req.get("system"), req.get("tools"), (req.get("messages") or [])[:upto]]),
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sign(req: dict[str, Any], thinking_text: str) -> str:
    """Signature for a thinking block generated as the reply to ``req``."""
    return f"mocksig.{prefix_hash(req, len(req.get('messages') or []))}.{_h(thinking_text)}"


def check_signatures(req: dict[str, Any]) -> str | None:
    """The API's error message for the first replayed thinking block that does not verify."""
    for i, m in enumerate(req.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for j, b in enumerate(_blocks(m.get("content"))):
            if b.get("type") != "thinking":
                continue
            sig = str(b.get("signature") or "")
            parts = sig.split(".")
            where = f"messages.{i}.content.{j}: Invalid `signature` in `thinking` block."
            if len(parts) != 3 or parts[0] != "mocksig" or parts[2] != _h(b.get("thinking", "")):
                return where
            if parts[1] != prefix_hash(req, i):
                return (
                    f"{where} The block is bound to a different conversation. Remove the "
                    'block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block".'
                )
    return None


def error_body(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_mock"}


def usage(req: dict[str, Any], n_prompt: int, cache_hit: int, n_out: int) -> dict[str, Any]:
    """Input tokens split the way the API reports them when ``cache_control`` is used."""
    cached = "cache_control" in json.dumps(req)
    if not cached:
        return {
            "input_tokens": n_prompt,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": n_out,
        }
    read = min(cache_hit, n_prompt)
    tail = min(3, n_prompt - read)  # tokens after the last breakpoint
    return {
        "input_tokens": tail,
        "cache_creation_input_tokens": n_prompt - read - tail,
        "cache_read_input_tokens": read,
        "output_tokens": n_out,
    }


def stop_reason(step: dict[str, Any], finish: str) -> str:
    if step.get("refusal"):
        return "refusal"
    return step.get("stop_reason") or STOP.get(finish, "end_turn")


def stop_details(step: dict[str, Any]) -> dict[str, Any] | None:
    if not step.get("refusal"):
        return None
    r = step["refusal"] if isinstance(step["refusal"], dict) else {}
    return {
        "type": "refusal",
        "category": r.get("category", "cyber"),
        "explanation": r.get("explanation", "The request was declined."),
    }


def group_blocks(pieces: list[tuple[str, Any]], show_thinking: bool) -> list[dict[str, Any]]:
    """Consecutive pieces → content blocks: {"type", "pieces", "tc"?}."""
    blocks: list[dict[str, Any]] = []
    for kind, piece in pieces:
        if kind == "tool":
            tc, text, _ = piece
            if not blocks or blocks[-1].get("tc") is not tc:
                blocks.append({"type": "tool_use", "tc": tc, "pieces": []})
            blocks[-1]["pieces"].append(text)
            continue
        btype = "thinking" if kind == "reasoning" else "text"
        if not blocks or blocks[-1]["type"] != btype:
            blocks.append({"type": btype, "pieces": []})
        blocks[-1]["pieces"].append(piece if (btype == "text" or show_thinking) else "")
    return blocks


def final_block(b: dict[str, Any], signature: str) -> dict[str, Any]:
    text = "".join(b["pieces"])
    if b["type"] == "thinking":
        return {"type": "thinking", "thinking": text, "signature": signature}
    if b["type"] == "text":
        return {"type": "text", "text": text}
    try:
        value = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        value = {}
    return {"type": "tool_use", "id": b["tc"]["id"], "name": b["tc"]["name"], "input": value}
