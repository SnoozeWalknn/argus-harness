"""Context compaction.

Triggered when the projected prompt passes ``threshold × n_ctx``:

1. ``mask``: tool output (and long tool-call arguments) older than the last
   ``keep_last_turns`` turns is replaced by one-line stubs. Cheap, and done in
   one batch so llama-server's prompt cache is invalidated rarely.
2. ``summarize``: if that is not enough, the middle of the history goes to a
   separate small model and is replaced by its summary.
3. ``drop``: if the small model is unavailable, the oldest turns are dropped.

The first two messages (system prompt, task) and the recent turns are never
touched, and a turn's tool calls always stay with their results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from argus.config import CompactionConfig
from argus.llm import LLMClient, LLMError
from argus.tokens import TokenCounter

HEAD = 2  # system prompt + task
TOOL_RESPONSE = "<tool_response>\n"
SUMMARY_PREFIX = "[Summary of earlier work; older messages were compacted]\n"
SUMMARY_SYSTEM = (
    "You compress the history of a coding agent's session so it can continue with less "
    "context. Write a concise summary covering: the task; what has been done (files read "
    "and changed, and how); key findings (errors, test results, relevant locations); what "
    "remains to do. Keep exact file paths, identifiers, commands and error messages. "
    "No preamble."
)
MAX_TRANSCRIPT_RESULT = 1500
LONG_ARG = 300

Entry = tuple[int, dict[str, Any]]  # (message id, message)


@dataclass
class Compaction:
    stage: str
    tokens_before: int
    tokens_after: int
    affected: int
    ms: float
    summary: str = ""
    model: str = ""
    error: str = ""


def message_text(msg: dict[str, Any]) -> str:
    text = msg.get("content") or ""
    if not isinstance(text, str):
        text = json.dumps(text)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        text += f"\n{fn.get('name')}({fn.get('arguments')})"
    return text


def turn_groups(context: list[Entry]) -> list[tuple[int, int]]:
    """[start, end) index ranges after the head; each starts at an assistant message."""
    groups: list[tuple[int, int]] = []
    start = HEAD
    for i in range(HEAD, len(context)):
        if context[i][1].get("role") == "assistant" and i > start:
            groups.append((start, i))
            start = i
    if start < len(context):
        groups.append((start, len(context)))
    return groups


def is_tool_result(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    return msg.get("role") == "tool" or (
        msg.get("role") == "user" and isinstance(content, str) and content.startswith(TOOL_RESPONSE)
    )


def _clip_strings(value: Any) -> Any:
    if isinstance(value, str) and len(value) > LONG_ARG:
        return value[:80] + f"…[{len(value) - 80} chars elided]"
    if isinstance(value, dict):
        return {k: _clip_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_strings(v) for v in value]
    return value


def mask_message(msg: dict[str, Any], min_chars: int) -> dict[str, Any] | None:
    """A shortened copy of an old message, or None if it is already small."""
    if is_tool_result(msg):
        content = msg.get("content") or ""
        if len(content) < min_chars:
            return None
        body = content.removeprefix(TOOL_RESPONSE).removesuffix("\n</tool_response>")
        lines = body.count("\n") + 1
        first = body.strip().splitlines()[0][:100] if body.strip() else ""
        stub = f"[old tool output elided to save context: {lines} lines, starting {first!r}; re-run the tool if needed]"
        if msg.get("role") == "user":
            stub = f"{TOOL_RESPONSE}{stub}\n</tool_response>"
        return {**msg, "content": stub}
    if msg.get("role") == "assistant":
        changed = False
        new = dict(msg)
        if msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if len(args) > min_chars:
                    try:
                        args = json.dumps(_clip_strings(json.loads(args)))
                        changed = True
                    except json.JSONDecodeError:
                        pass
                calls.append({**tc, "function": {**fn, "arguments": args}})
            new["tool_calls"] = calls
        content = msg.get("content") or ""
        if len(content) > min_chars and content.lstrip().startswith("{"):
            try:  # constrained protocols: the action JSON is the content
                new["content"] = json.dumps(
                    _clip_strings(json.loads(content)), separators=(",", ":")
                )
                changed = True
            except json.JSONDecodeError:
                pass
        new.pop("reasoning_content", None)
        return new if changed or "reasoning_content" in msg else None
    return None


def transcript(context: list[Entry], lo: int, hi: int) -> str:
    out = []
    for _, msg in context[lo:hi]:
        role = "result" if is_tool_result(msg) else msg.get("role")
        text = message_text(msg).removeprefix(TOOL_RESPONSE)
        if role == "result" and len(text) > MAX_TRANSCRIPT_RESULT:
            text = (
                text[:MAX_TRANSCRIPT_RESULT] + f"\n…[{len(text) - MAX_TRANSCRIPT_RESULT} chars cut]"
            )
        out.append(f"## {role}\n{text.strip()}")
    return "\n\n".join(out)


class Compactor:
    def __init__(self, cfg: CompactionConfig, counter: TokenCounter):
        self.cfg = cfg
        self.counter = counter
        self.llm: LLMClient | None = None
        if cfg.summarize:
            self.llm = LLMClient(cfg.base_url, cfg.api_key, cfg.timeout, 5.0, retries=0)

    def close(self) -> None:
        if self.llm:
            self.llm.close()

    def tokens(self, msg: dict[str, Any]) -> int:
        return self.counter.count(message_text(msg)) + 4  # role framing

    def limit(self, n_ctx: int) -> int:
        return int(self.cfg.threshold * n_ctx)

    def mask(
        self, context: list[Entry], min_chars: int = 400
    ) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        """Masked replacements [(index, new message)] for turns before the recent ones; tokens saved."""
        groups = turn_groups(context)
        old = groups[: max(len(groups) - self.cfg.keep_last_turns, 0)]
        out, saved = [], 0
        for lo, hi in old:
            for i in range(lo, hi):
                new = mask_message(context[i][1], min_chars)
                if new is not None:
                    saved += self.tokens(context[i][1]) - self.tokens(new)
                    out.append((i, new))
        return out, saved

    def summarize(self, task: str, context: list[Entry], hi: int) -> str:
        assert self.llm is not None
        text = transcript(context, HEAD, hi)
        budget = 48_000
        if len(text) > budget:
            text = text[: budget // 3] + "\n\n[...]\n\n" + text[-2 * budget // 3 :]
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": f"Task:\n{task}\n\nHistory:\n{text}"},
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        c = self.llm.chat(body, stream=False)
        summary = c.content.strip()
        if not summary:
            raise LLMError("compaction model returned an empty summary")
        return summary

    def middle_end(self, context: list[Entry]) -> int:
        """Index where the kept recent turns begin."""
        groups = turn_groups(context)
        keep = groups[-self.cfg.keep_last_turns :] if self.cfg.keep_last_turns else []
        return keep[0][0] if keep else len(context)


def timed() -> float:
    return time.perf_counter()
