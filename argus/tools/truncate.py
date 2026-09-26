"""Shrinking command output for the model."""

from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def head_tail(text: str, max_chars: int, head_lines: int, tail_lines: int) -> tuple[str, bool]:
    """Keep the first and last lines when output is too long."""
    if len(text) <= max_chars:
        return text, False
    lines = text.splitlines()
    if len(lines) > head_lines + tail_lines:
        omitted = len(lines) - head_lines - tail_lines
        lines = lines[:head_lines] + [f"[... {omitted} lines omitted ...]"] + lines[-tail_lines:]
    out = "\n".join(lines)
    if len(out) > max_chars:
        half = max_chars // 2
        out = f"{out[:half]}\n[... {len(out) - 2 * half} chars omitted ...]\n{out[-half:]}"
    return out, True
