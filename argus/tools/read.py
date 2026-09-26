from __future__ import annotations

import difflib
import posixpath

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj

MAX_READ_BYTES = 10_000_000


def number_lines(lines: list[str], start: int, max_chars: int) -> str:
    out = []
    for n, line in enumerate(lines, start):
        if len(line) > max_chars:
            line = f"{line[:max_chars]}…[+{len(line) - max_chars} chars]"
        out.append(f"{n}\t{line}")
    return "\n".join(out)


def looks_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def not_found(ctx: ToolContext, path: str) -> ToolError:
    """File-not-found error with nearby suggestions, which saves the model a glob call."""
    msg = f"file not found: {ctx.executor.rel(path)}"
    try:
        files = ctx.executor.list_files(timeout=10)
    except Exception:  # suggestions are best effort
        files = []
    want = ctx.executor.rel(path)
    base = posixpath.basename(want)
    same_name = [f for f in files if posixpath.basename(f) == base][:5]
    close = same_name or difflib.get_close_matches(want, files, n=3, cutoff=0.6)
    if close:
        msg += "; did you mean: " + ", ".join(close)
    return ToolError(msg)


def read_text(ctx: ToolContext, path: str) -> tuple[str, bytes]:
    """Read a file as text, raising ToolError with a useful message on failure."""
    ex = ctx.executor
    try:
        data = ex.read_bytes(path, max_bytes=MAX_READ_BYTES + 1)
    except FileNotFoundError:
        raise not_found(ctx, path) from None
    except IsADirectoryError:
        raise ToolError(f"{ex.rel(path)} is a directory; use glob to list files") from None
    except PermissionError:
        raise ToolError(f"permission denied: {ex.rel(path)}") from None
    if len(data) > MAX_READ_BYTES:
        raise ToolError(f"{ex.rel(path)} is larger than {MAX_READ_BYTES} bytes; use grep or bash")
    if looks_binary(data):
        raise ToolError(f"{ex.rel(path)} is a binary file ({len(data)} bytes)")
    return data.decode("utf-8", "replace"), data


class ReadTool(Tool):
    name = "read"
    description = "Read a text file. Lines are prefixed with their line number and a tab."
    parameters = obj(
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "first line (1-based)"},
            "limit": {"type": "integer", "description": "max lines"},
        },
        ["path"],
    )
    signature = "read(path, offset?, limit?) - show file with line numbers"

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        path = ctx.executor.resolve(args["path"])
        text, data = read_text(ctx, path)
        ctx.tracker.mark(path, data)
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult("[empty file]")
        offset = max(int(args.get("offset") or 1), 1)
        if offset > total:
            raise ToolError(f"offset {offset} is past the end ({total} lines)")
        limit = int(args.get("limit") or ctx.cfg.read_max_lines)
        limit = max(1, min(limit, ctx.cfg.read_max_lines))
        chunk = lines[offset - 1 : offset - 1 + limit]
        body = number_lines(chunk, offset, ctx.cfg.read_max_line_chars)
        end = offset + len(chunk) - 1
        truncated = offset > 1 or end < total
        if truncated:
            more = f"; offset={end + 1} for more" if end < total else ""
            body += f"\n[lines {offset}-{end} of {total}{more}]"
        return ToolResult(body, truncated=truncated, meta={"path": path, "lines": total})
