"""Locating `old` text that is almost, but not exactly, in the file.

The cascade, from safest to loosest:

1. line-number prefixes copied from ``read`` output are removed from ``old``
2. line match ignoring trailing whitespace
3. line match ignoring indentation (``new`` is re-indented to fit)
4. line match after unicode normalisation (smart quotes, dashes, nbsp)
5. whitespace-flexible match inside a line
6. similarity match over line windows, applied only above a threshold and
   only when the best candidate clearly beats the runner-up

Steps 1–5 never change meaning for code that parses the same; step 6 is
reported to the model so it can double-check.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

MAX_SIMILARITY_LINES = 20_000
TOKEN = re.compile(r"\w+|[^\w\s]")
WORD = re.compile(r"\w")
LINE_NO = re.compile(r"^\s*\d+(?:\t|[|:→] ?)")
UNICODE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "…": "...",
        "​": "",
    }
)


@dataclass
class Span:
    start: int  # char offsets in the file text
    end: int
    replacement: str


@dataclass
class FuzzyResult:
    spans: list[Span]
    how: str  # human-readable description of the step that matched


@dataclass
class Closest:
    first_line: int  # 1-based
    last_line: int
    ratio: float
    diff: str


class Ambiguous(Exception):
    def __init__(self, how: str, lines: list[int]):
        super().__init__(how)
        self.how = how
        self.lines = lines


def strip_line_numbers(text: str) -> str | None:
    """Remove ``12<tab>`` style prefixes if every non-blank line has one."""
    lines = text.split("\n")
    body = [ln for ln in lines if ln.strip()]
    if not body or not all(LINE_NO.match(ln) for ln in body):
        return None
    return "\n".join(LINE_NO.sub("", ln, count=1) if ln.strip() else ln for ln in lines)


def trim_like(old_lines: list[str], new_lines: list[str]) -> tuple[list[str], list[str]]:
    """Drop blank lines at the edges of ``old``, and the matching blank edges of ``new``."""
    new_lines = list(new_lines)
    lead = 0
    while lead < len(old_lines) and not old_lines[lead].strip():
        lead += 1
    trail = 0
    while trail < len(old_lines) - lead and not old_lines[len(old_lines) - 1 - trail].strip():
        trail += 1
    for _ in range(lead):
        if new_lines and not new_lines[0].strip():
            new_lines.pop(0)
    for _ in range(trail):
        if new_lines and not new_lines[-1].strip():
            new_lines.pop()
    return old_lines[lead : len(old_lines) - trail], new_lines


def _trim_blank_edges(lines: list[str]) -> list[str]:
    i, j = 0, len(lines)
    while i < j and not lines[i].strip():
        i += 1
    while j > i and not lines[j - 1].strip():
        j -= 1
    return lines[i:j]


def _line_offsets(text: str) -> list[int]:
    offs = [0]
    for m in re.finditer("\n", text):
        offs.append(m.end())
    return offs


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _common_indent(lines: list[str]) -> str:
    ind = [_indent(ln) for ln in lines if ln.strip()]
    if not ind:
        return ""
    prefix = ind[0]
    for i in ind[1:]:
        while not i.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def reindent(new_lines: list[str], old_indent: str, file_indent: str) -> list[str]:
    out = []
    for ln in new_lines:
        if not ln.strip():
            out.append(ln)
        elif ln.startswith(old_indent):
            out.append(file_indent + ln[len(old_indent) :])
        else:
            out.append(file_indent + ln.lstrip())
    return out


class Matcher:
    def __init__(self, text: str, threshold: float):
        self.text = text
        self.lines = text.split("\n")
        self.offsets = _line_offsets(text)
        self.threshold = threshold

    def _span(self, first: int, count: int, replacement_lines: list[str]) -> Span:
        start = self.offsets[first]
        last = first + count - 1
        end = self.offsets[last] + len(self.lines[last])
        return Span(start, end, "\n".join(replacement_lines))

    def _windows(self, old_lines: list[str], key) -> list[int]:
        want = [key(ln) for ln in old_lines]
        n = len(want)
        keyed = [key(ln) for ln in self.lines]
        return [i for i in range(len(keyed) - n + 1) if keyed[i : i + n] == want]

    def find(self, old: str, new: str, replace_all: bool) -> FuzzyResult | None:
        old_lines, new_lines = trim_like(old.split("\n"), new.split("\n"))
        if not old_lines:
            return None
        steps = [
            ("ignoring trailing whitespace", lambda s: s.rstrip(), False),
            ("ignoring indentation", lambda s: s.strip(), True),
            (
                "after normalising unicode punctuation",
                lambda s: s.translate(UNICODE_MAP).strip(),
                True,
            ),
        ]
        for how, key, fix_indent in steps:
            hits = self._windows(old_lines, key)
            if not hits:
                continue
            if len(hits) > 1 and not replace_all:
                raise Ambiguous(how, [h + 1 for h in hits])
            spans = []
            for h in hits:
                repl = new_lines
                if fix_indent:
                    window = self.lines[h : h + len(old_lines)]
                    repl = reindent(new_lines, _common_indent(old_lines), _common_indent(window))
                spans.append(self._span(h, len(old_lines), repl))
            return FuzzyResult(spans, how)
        if len(old_lines) == 1:
            found = self._inline(old_lines[0].strip(), new, replace_all)
            if found:
                return found
        return self._similar(old_lines, new_lines)

    def _inline(self, needle: str, new: str, replace_all: bool) -> FuzzyResult | None:
        """Match within a line ignoring spacing between tokens (``a*b`` vs ``a * b``)."""
        tokens = TOKEN.findall(needle)
        if len(tokens) < 2:
            return None
        parts = [re.escape(tokens[0])]
        for prev, tok in zip(tokens, tokens[1:], strict=False):
            # Two words need some space between them; anything next to punctuation may have none.
            parts.append(r"[ \t]+" if WORD.match(prev[-1]) and WORD.match(tok[0]) else r"[ \t]*")
            parts.append(re.escape(tok))
        hits = list(re.finditer("".join(parts), self.text))
        if not hits:
            return None
        if len(hits) > 1 and not replace_all:
            raise Ambiguous(
                "ignoring spacing", [self.text.count("\n", 0, m.start()) + 1 for m in hits]
            )
        return FuzzyResult([Span(m.start(), m.end(), new) for m in hits], "ignoring spacing")

    def ranked(self, old_lines: list[str]) -> list[tuple[float, int, int]]:
        """(ratio, first index, length) of candidate windows, best first."""
        n = len(old_lines)
        if len(self.lines) > MAX_SIMILARITY_LINES:
            return []
        target = "\n".join(ln.strip() for ln in old_lines)
        stripped = [ln.strip() for ln in self.lines]
        out = []
        for size in {n, max(n - 1, 1), n + 1}:
            for i in range(0, max(len(self.lines) - size + 1, 0)):
                cand = "\n".join(stripped[i : i + size])
                sm = difflib.SequenceMatcher(None, target, cand, autojunk=False)
                if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
                    continue
                out.append((sm.ratio(), i, size))
        out.sort(key=lambda t: -t[0])
        return out

    def _similar(self, old_lines: list[str], new_lines: list[str]) -> FuzzyResult | None:
        if self.threshold > 1:
            return None
        ranked = self.ranked(old_lines)
        if not ranked:
            return None
        best, i, size = ranked[0]
        # Runner-up must be a different place, not an overlapping window of the same one.
        rivals = [r for r, j, s in ranked[1:] if j + s <= i or j >= i + size]
        second = rivals[0] if rivals else 0.0
        if best < self.threshold or best - second < 0.05:
            return None
        window = self.lines[i : i + size]
        repl = reindent(new_lines, _common_indent(old_lines), _common_indent(window))
        return FuzzyResult(
            [self._span(i, size, repl)], f"by similarity ({best:.0%}) at lines {i + 1}-{i + size}"
        )

    def closest(self, old: str) -> Closest | None:
        old_lines = _trim_blank_edges(old.split("\n"))
        if not old_lines:
            return None
        ranked = self.ranked(old_lines)
        if not ranked:
            return None
        ratio, i, size = ranked[0]
        window = self.lines[i : i + size]
        diff = "\n".join(
            ln
            for ln in difflib.unified_diff(old_lines, window, "your old", "file", lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))
        )
        return Closest(i + 1, i + size, ratio, diff)


def apply_spans(text: str, spans: list[Span]) -> str:
    for sp in sorted(spans, key=lambda s: s.start, reverse=True):
        text = text[: sp.start] + sp.replacement + text[sp.end :]
    return text
