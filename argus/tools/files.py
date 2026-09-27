"""Whole-file writes and directory listings."""

from __future__ import annotations

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, digest, obj
from argus.tools.edit import syntax_warning
from argus.tools.read import read_text

LS_MAX_ENTRIES = 300


class WriteTool(Tool):
    name = "write"
    description = (
        "Create a file, or replace a whole file's content. An existing file must be read "
        "first. For changes to part of a file use edit."
    )
    parameters = obj(
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]
    )
    summary = "create a file or replace all of it (read it first if it exists)"
    mutating = True

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        ex = ctx.executor
        path = ex.resolve(args["path"])
        rel = ex.rel(path)
        content = args["content"]
        if not ctx.cfg.allow_write_outside_workdir and not ex.inside_workdir(path):
            raise ToolError(f"{path} is outside the workspace {ex.workdir}")
        existed = ex.exists(path)
        if existed:
            _, data = read_text(ctx, path)
            if ctx.cfg.require_read_before_edit and not ctx.tracker.seen(path):
                raise ToolError(f"read {rel} before replacing it")
            if ctx.tracker.is_stale(path, data):
                raise ToolError(f"{rel} changed since you read it; read it again first")
        out = content.encode()
        ex.write_bytes(path, out)
        ctx.tracker.mark(path, out)
        n = len(content.splitlines())
        msg = f"{'Replaced' if existed else 'Created'} {rel} ({n} lines)."
        warn = syntax_warning(rel, content)
        text = f"{msg}\n{warn}" if warn else msg
        return ToolResult(
            text,
            meta={
                "path": path,
                "sha1": digest(out),
                "created": not existed,
                "syntax_warning": warn,
            },
        )


class LsTool(Tool):
    name = "ls"
    description = (
        "Show the file tree under a directory (default: the workspace root), honouring "
        ".gitignore. Directories deeper than `depth` are summarised with a file count."
    )
    parameters = obj(
        {
            "path": {"type": "string", "description": "directory"},
            "depth": {"type": "integer", "description": "levels to expand (default 2)"},
        },
        [],
    )
    summary = "file tree of a directory"

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        ex = ctx.executor
        target = args.get("path") or "."
        if not ex.inside_workdir(target):
            raise ToolError(f"{target} is outside the workspace; use bash (ls) there")
        rel = ex.rel(target)
        prefix = "" if rel == "." else rel.rstrip("/") + "/"
        depth = min(max(int(args.get("depth") or 2), 1), 6)
        files = [f[len(prefix) :] for f in ex.list_files() if f.startswith(prefix)]
        if not files:
            if prefix and ex.exists(target):
                raise ToolError(f"{rel} is a file or an empty directory; read it or use bash")
            raise ToolError(f"no files under {rel}")
        tree: dict = {}
        for f in files:
            node = tree
            for part in f.split("/")[:-1]:
                node = node.setdefault(part + "/", {})
            node[f.rsplit("/", 1)[-1]] = None
        lines: list[str] = [f"{prefix or './'}  ({len(files)} files)"]
        self._render(tree, 1, depth, lines)
        truncated = len(lines) > LS_MAX_ENTRIES
        if truncated:
            lines = lines[:LS_MAX_ENTRIES] + [f"[{len(lines) - LS_MAX_ENTRIES} more entries]"]
        return ToolResult("\n".join(lines), truncated=truncated, meta={"count": len(files)})

    def _render(self, node: dict, level: int, depth: int, out: list[str]) -> None:
        dirs = sorted(k for k in node if k.endswith("/"))
        files = sorted(k for k in node if not k.endswith("/"))
        pad = "  " * level
        for d in dirs:
            if level >= depth:
                out.append(f"{pad}{d}  ({_count(node[d])} files)")
            else:
                out.append(f"{pad}{d}")
                self._render(node[d], level + 1, depth, out)
        out.extend(f"{pad}{f}" for f in files)


def _count(node: dict) -> int:
    return sum(1 if v is None else _count(v) for v in node.values())
