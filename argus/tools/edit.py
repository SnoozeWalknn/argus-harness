from __future__ import annotations

import json

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj
from argus.tools.read import number_lines, read_text

SNIPPET_CONTEXT = 3
SNIPPET_MAX_LINES = 30


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def find_all(text: str, needle: str) -> list[int]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + max(len(needle), 1))
    return out


def snippet(text: str, first_line: int, last_line: int, max_chars: int) -> str:
    lines = text.splitlines()
    lo = max(first_line - SNIPPET_CONTEXT, 1)
    hi = min(last_line + SNIPPET_CONTEXT, len(lines))
    if hi - lo + 1 > SNIPPET_MAX_LINES:
        hi = lo + SNIPPET_MAX_LINES - 1
    return number_lines(lines[lo - 1 : hi], lo, max_chars)


def syntax_warning(path: str, text: str) -> str | None:
    """Cheap post-edit sanity check for languages we can parse in-process."""
    if path.endswith(".py"):
        try:
            compile(text, path, "exec", dont_inherit=True)
        except SyntaxError as e:
            return f"warning: {path} now has a Python syntax error at line {e.lineno}: {e.msg}"
        except ValueError:
            return None
    elif path.endswith(".json"):
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            return f"warning: {path} is no longer valid JSON (line {e.lineno}: {e.msg})"
    return None


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace exact text in a file. `old` must match exactly once unless all=true. "
        "Empty `old` creates a new file."
    )
    parameters = obj(
        {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "all": {"type": "boolean", "description": "replace every occurrence"},
        },
        ["path", "old", "new"],
    )
    summary = 'replace exact text (unique unless all); old="" creates a file'
    mutating = True

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        ex = ctx.executor
        path = ex.resolve(args["path"])
        rel = ex.rel(path)
        old, new = args["old"], args["new"]
        if not ctx.cfg.allow_write_outside_workdir and not ex.inside_workdir(path):
            raise ToolError(f"{path} is outside the workspace {ex.workdir}")

        if old == "":
            if ex.exists(path):
                raise ToolError(
                    f"{rel} already exists; `old` must be text from the file (empty only creates files)"
                )
            data = new.encode()
            ex.write_bytes(path, data)
            ctx.tracker.mark(path, data)
            n = len(new.splitlines())
            return self._done(ctx, rel, path, new, f"Created {rel} ({n} lines).", None)

        if not ex.exists(path):
            raise ToolError(f'{rel} does not exist; to create it use old=""')
        text, data = read_text(ctx, path)
        if ctx.cfg.require_read_before_edit and not ctx.tracker.seen(path):
            raise ToolError(f"read {rel} before editing it")
        if old == new:
            raise ToolError("`old` and `new` are identical; nothing to change")

        new_text, first, last, count, note = self.apply(
            ctx, rel, text, old, new, bool(args.get("all"))
        )
        if new_text == text:
            raise ToolError("edit produced no change")
        out = new_text.encode()
        ex.write_bytes(path, out)
        ctx.tracker.mark(path, out)
        what = f"{count} occurrences" if count > 1 else f"lines {first}-{last}"
        msg = f"Edited {rel} ({what})."
        if note:
            msg += f" {note}"
        return self._done(ctx, rel, path, new_text, msg, (first, last))

    def apply(
        self, ctx: ToolContext, rel: str, text: str, old: str, new: str, replace_all: bool
    ) -> tuple[str, int, int, int, str]:
        """Return (new_text, first_line, last_line, count, note) or raise ToolError."""
        hits = find_all(text, old)
        if len(hits) == 1 or (hits and replace_all):
            first = line_of(text, hits[0])
            new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            last = first + max(len(new.splitlines()), 1) - 1
            return new_text, first, last, len(hits), ""
        if len(hits) > 1:
            lines = ", ".join(str(line_of(text, h)) for h in hits[:10])
            raise ToolError(
                f"`old` matches {len(hits)} times in {rel} (lines {lines}); "
                "add surrounding lines to make it unique, or set all=true"
            )
        raise ToolError(self.no_match_message(ctx, rel, text, old))

    def no_match_message(self, ctx: ToolContext, rel: str, text: str, old: str) -> str:
        return f"`old` not found in {rel}; read the file and copy the text exactly"

    def _done(
        self,
        ctx: ToolContext,
        rel: str,
        path: str,
        text: str,
        msg: str,
        span: tuple[int, int] | None,
    ) -> ToolResult:
        parts = [msg]
        if span:
            parts.append(snippet(text, span[0], span[1], ctx.cfg.read_max_line_chars))
        warn = syntax_warning(rel, text)
        if warn:
            parts.append(warn)
        return ToolResult("\n".join(parts), meta={"path": path, "syntax_warning": warn})
