"""Sessions: continue a conversation across runs, on the same model or another one.

A session is a sequence of runs sharing one conversation. Each run stays a
complete row of its own; at its end argus saves the context it finished with,
and the next run of the session starts from that context plus the new message.

The next run may use another model, provider or protocol (OpenCode-style
switching). Its system prompt replaces the old one, and:

* when the tool-call protocol family changes, history is converted: JSON
  actions and ``<tool_response>`` messages (constrained protocols) become native
  tool calls and tool results, or the other way round;
* when the system prompt or the model changed, the history was effectively
  edited, so replayed reasoning (thinking signatures, encrypted reasoning) is
  stripped; providers bind it to the exact conversation that produced it.
"""

from __future__ import annotations

import json
from typing import Any

from argus.compact import TOOL_RESPONSE

NATIVE = "native"
Entry = tuple[int, dict[str, Any]]


def family(protocol: str) -> str:
    return NATIVE if protocol == NATIVE else "constrained"


def _envelope(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
        return obj
    return None


def _tool_response(content: Any) -> str | None:
    if isinstance(content, str) and content.startswith(TOOL_RESPONSE):
        return content.removeprefix(TOOL_RESPONSE).removesuffix("\n</tool_response>")
    return None


def to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Constrained-protocol history → native tool calls and tool results."""
    out: list[dict[str, Any]] = []
    pending: str | None = None  # id of the call whose result comes next
    for i, m in enumerate(messages):
        env = _envelope(m.get("content")) if m.get("role") == "assistant" else None
        if env is not None and not m.get("tool_calls"):
            if env["tool"] == "done":
                args = env.get("args") or {}
                out.append({"role": "assistant", "content": str(args.get("summary", ""))})
                continue
            pending = f"hist_{i}"
            args = env.get("args") if isinstance(env.get("args"), dict) else {}
            out.append(
                {
                    "role": "assistant",
                    "content": str(env.get("thought", "")),
                    "tool_calls": [
                        {
                            "id": pending,
                            "type": "function",
                            "function": {"name": env["tool"], "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            continue
        result = _tool_response(m.get("content")) if m.get("role") == "user" else None
        if result is not None and pending is not None:
            out.append({"role": "tool", "tool_call_id": pending, "content": result})
            pending = None
            continue
        out.append(m)
    return out


def to_constrained(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Native history → one JSON action per assistant turn and ``<tool_response>`` messages."""
    results = {
        m.get("tool_call_id"): m.get("content", "") for m in messages if m.get("role") == "tool"
    }
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            continue  # emitted right after its call below
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                action = {"tool": fn.get("name", ""), "args": args}
                out.append({"role": "assistant", "content": json.dumps(action)})
                text = results.get(tc.get("id"), "")
                out.append({"role": "user", "content": f"{TOOL_RESPONSE}{text}\n</tool_response>"})
            continue
        out.append(
            {k: v for k, v in m.items() if k != "replay"} if m.get("role") == "assistant" else m
        )
    return out


def convert(
    messages: list[dict[str, Any]], from_protocol: str, to_protocol: str
) -> list[dict[str, Any]]:
    if family(from_protocol) == family(to_protocol):
        return messages
    return to_native(messages) if family(to_protocol) == NATIVE else to_constrained(messages)


def history(
    entries: list[Entry], *, from_protocol: str, to_protocol: str
) -> tuple[list[Entry], bool]:
    """The previous context without its system prompt, converted for ``to_protocol``.

    Returns (entries, converted); converted messages get id 0 (they need new rows).
    """
    body = [(mid, msg) for mid, msg in entries if msg.get("role") != "system"]
    msgs = [msg for _, msg in body]
    new = convert(msgs, from_protocol, to_protocol)
    if new is msgs:
        return body, False
    return [(0, m) for m in new], True
