"""Shrinking command output for the model.

``smart_truncate`` cleans terminal noise first (ANSI codes, carriage-return
progress bars, runs of identical lines), and only then cuts: it keeps the
head and the tail, where commands put their summary and errors, plus any
error-looking lines from the elided middle.
"""

from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
NOTABLE_RE = re.compile(
    r"error|fail|exception|traceback|panic|fatal|assert|warning|denied|not found|undefined|"
    r"cannot|can't|unable|segfault|killed",
    re.I,
)
MAX_NOTABLE = 20


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def collapse_carriage_returns(text: str) -> str:
    """Keep only the final state of lines rewritten with \\r (progress bars, spinners)."""
    text = text.replace("\r\n", "\n")
    if "\r" not in text:
        return text
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


def fold_repeats(lines: list[str], min_run: int = 3) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            out.append(lines[i])
            out.append(f"[... previous line repeated {run - 1} more times]")
        else:
            out.extend(lines[i : j + 1])
        i = j + 1
    return out


def clip_line(line: str, width: int) -> str:
    return line if len(line) <= width else f"{line[:width]}…[+{len(line) - width} chars]"


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


def smart_truncate(
    text: str, max_chars: int, head_lines: int, tail_lines: int, line_width: int
) -> tuple[str, bool]:
    """Clean and shrink output. Returns (text, truncated)."""
    raw_lines = collapse_carriage_returns(strip_ansi(text)).split("\n")
    lines = fold_repeats(raw_lines)
    cleaned = "\n".join(lines)
    if len(cleaned) <= max_chars:
        return cleaned, len(lines) != len(raw_lines)
    lines = [clip_line(ln, line_width) for ln in lines]
    if len(lines) > head_lines + tail_lines:
        middle = lines[head_lines : len(lines) - tail_lines]
        notable = [(i + head_lines + 1, ln) for i, ln in enumerate(middle) if NOTABLE_RE.search(ln)]
        kept = notable[:MAX_NOTABLE]
        marker = f"[... {len(middle)} lines omitted"
        if kept:
            marker += f"; {len(kept)} notable lines kept below with their line numbers"
        marker += " ...]"
        body = lines[:head_lines] + [marker] + [f"{n}: {ln}" for n, ln in kept]
        if kept:
            body.append("[... end of notable lines ...]")
        lines = body + lines[-tail_lines:]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out, _ = head_tail(out, max_chars, head_lines, tail_lines)
    return out, True
