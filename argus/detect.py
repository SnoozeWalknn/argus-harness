"""Failure detectors and stream monitors.

Tags (``failures.tag``):

``loop``            the same call keeps producing the same result, a short cycle of such
                    calls repeats, or a generation degenerates into repeating text
``overrun``         the model keeps going after it is done: text past the final answer
                    (role markers, a repeated summary), or more tool turns after it
                    declared the task complete
``malformed_call``  a call or reply that could not be parsed or validated
``token_cap``       a token limit cut the model off: max_tokens, the reasoning budget,
                    the context window or the run's token budget
``refusal``         the provider declined the request (safety stop), with its reason
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from argus.providers.base import Delta, StopGeneration

# -- loops ---------------------------------------------------------------------------------------


def call_key(name: str, args: dict[str, Any], result: str) -> str:
    blob = json.dumps([name, args, result], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


@dataclass
class LoopVerdict:
    count: int  # repetitions observed
    detail: str
    abort: bool
    key: str  # the (call, result) that triggered it


class LoopDetector:
    """Watches (call, result) pairs. Repeating a call is only a loop if nothing changed."""

    def __init__(self, repeat: int = 3, abort: int = 5, window: int = 12):
        self.repeat = max(repeat, 2)
        self.abort_at = max(abort, self.repeat)
        self.window = window
        self.history: list[str] = []
        self.labels: dict[str, str] = {}

    def observe(
        self, name: str, args: dict[str, Any], result: str, label: str
    ) -> LoopVerdict | None:
        key = call_key(name, args, result)
        self.history.append(key)
        self.labels[key] = label
        h = self.history

        run = 1  # identical consecutive calls
        while run < len(h) and h[-1 - run] == key:
            run += 1
        recent = h[-self.window :]
        freq = recent.count(key)
        cycle = self._cycle()

        count, detail = 0, ""
        if run >= self.repeat:
            count, detail = run, f"same call and result {run}x in a row: {label}"
        elif cycle:
            period, reps = cycle
            count = reps
            labels = " → ".join(self.labels[k] for k in h[-period:])
            detail = f"cycle of {period} calls repeated {reps}x: {labels}"
        elif freq >= self.repeat + 1:
            count, detail = (
                freq - 1,
                f"same call and result {freq}x in the last {len(recent)} calls: {label}",
            )
        if not count:
            return None
        return LoopVerdict(count, detail, abort=count >= self.abort_at, key=key)

    def _cycle(self) -> tuple[int, int] | None:
        """(period, repetitions) if the history ends in a repeating cycle of distinct calls."""
        h = self.history
        for period in (2, 3):
            unit = h[-period:]
            if len(unit) < period or len(set(unit)) != period:
                continue
            reps = 1
            while (
                len(h) >= period * (reps + 1)
                and h[len(h) - period * (reps + 1) : len(h) - period * reps] == unit
            ):
                reps += 1
            if reps >= self.repeat:
                return period, reps
        return None


def periodic_tail(text: str, window: int, min_period: int = 8, min_reps: int = 4) -> str | None:
    """The repeating unit if the last ``window`` chars are one unit repeated ``min_reps``+ times."""
    if window <= 0 or len(text) < window:
        return None
    tail = text[-window:]
    for p in range(min_period, window // min_reps + 1):
        if tail[p:] == tail[:-p]:
            return tail[-p:]
    return None


# -- stream monitors -----------------------------------------------------------------------------


class ReasoningBudget:
    def __init__(self, max_tokens: int):
        self.max = max_tokens

    def __call__(self, d: Delta) -> None:
        if d.kind == "reasoning" and d.completion.reasoning_tokens > self.max:
            raise StopGeneration("reasoning_budget", f"reasoning exceeded {self.max} tokens")


class RepetitionMonitor:
    def __init__(self, window: int = 400, every: int = 32):
        self.window = window
        self.every = every
        self.n = 0

    def __call__(self, d: Delta) -> None:
        if d.kind == "tool":  # file contents in edit args may legitimately repeat
            return
        self.n += 1
        if self.n % self.every:
            return
        text = d.completion.reasoning if d.kind == "reasoning" else d.completion.content
        unit = periodic_tail(text, self.window)
        if unit:
            raise StopGeneration("repetition", f"{d.kind} degenerated into repeating {unit[:60]!r}")


# -- overrun -------------------------------------------------------------------------------------

ROLE_MARKER = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<tool_response>|<\|user\|>"
    r"|^\s*(?:user|User|USER|Human|HUMAN)\s*:",
    re.M,
)
COMPLETION_CLAIM = re.compile(
    r"\b(?:the\s+)?(?:task|work|fix|implementation)\s+(?:is|has\s+been)\s+(?:now\s+)?"
    r"(?:complete|completed|done|finished)\b"
    r"|\ball\s+(?:the\s+)?tests\s+(?:now\s+)?pass"
    r"|\bI\s+have\s+(?:successfully\s+)?(?:completed|finished)\s+the\s+task"
    r"|\beverything\s+(?:is\s+)?(?:now\s+)?working\b",
    re.I,
)


def cut_at_role_marker(text: str) -> tuple[str, str | None]:
    """Cut hallucinated continuation (a fake next turn) off a final answer."""
    m = ROLE_MARKER.search(text)
    if not m or not text[: m.start()].strip():
        return text, None
    return text[: m.start()].rstrip(), m.group(0).strip()


def repeated_paragraph(text: str, min_chars: int = 40) -> str | None:
    seen: set[str] = set()
    for para in re.split(r"\n\s*\n", text):
        p = " ".join(para.split())
        if len(p) < min_chars:
            continue
        if p in seen:
            return p
        seen.add(p)
    return None


def claims_completion(text: str) -> bool:
    return bool(COMPLETION_CLAIM.search(text or ""))
