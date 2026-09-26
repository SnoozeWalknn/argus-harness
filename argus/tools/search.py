from __future__ import annotations

import re
import shlex

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj


def _translate(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    depth = 0
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                j = i + 2
                if j < n and pat[j] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pat[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j
        elif c == "{":
            out.append("(?:")
            depth += 1
        elif c == "}" and depth:
            out.append(")")
            depth -= 1
        elif c == "," and depth:
            out.append("|")
        else:
            out.append(re.escape(c))
        i += 1
    out.append(")" * depth)
    return "".join(out)


def glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob. ``**`` spans directories; a pattern without ``/`` matches basenames."""
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    body = _translate(pattern)
    if "/" not in pattern:
        body = f"(?:.*/)?{body}"
    return re.compile(f"^{body}$")


def _scope(ctx: ToolContext, path: str | None) -> str:
    """Relative directory prefix for a user-supplied path ('' = workspace root)."""
    if not path or path.strip() in (".", "./", ""):
        return ""
    ex = ctx.executor
    if not ex.inside_workdir(path):
        raise ToolError(f"{path} is outside the workspace; use bash to search there")
    rel = ex.rel(path)
    return "" if rel == "." else rel.rstrip("/") + "/"


class GlobTool(Tool):
    name = "glob"
    description = "List files matching a glob pattern such as **/*.py or src/*.ts."
    parameters = obj(
        {"pattern": {"type": "string"}, "path": {"type": "string", "description": "subdirectory"}},
        ["pattern"],
    )
    signature = "glob(pattern, path?) - list files matching a glob"

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        prefix = _scope(ctx, args.get("path"))
        rx = glob_regex(args["pattern"])
        files = ctx.executor.list_files()
        hits = [f for f in files if f.startswith(prefix) and rx.match(f[len(prefix) :])]
        if not hits:
            return ToolResult("No files matched.", meta={"count": 0})
        limit = ctx.cfg.glob_max_results
        body = "\n".join(hits[:limit])
        if len(hits) > limit:
            body += f"\n[{len(hits) - limit} more; narrow the pattern]"
        return ToolResult(body, truncated=len(hits) > limit, meta={"count": len(hits)})


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents with a regular expression. Returns path:line:text."
    parameters = obj(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "file or directory"},
            "glob": {"type": "string", "description": "only files matching this glob"},
            "ignore_case": {"type": "boolean"},
        },
        ["pattern"],
    )
    signature = "grep(pattern, path?, glob?, ignore_case?) - regex search, returns path:line:text"

    def command(self, ctx: ToolContext, args: dict) -> str:
        pattern = args["pattern"]
        target = args.get("path") or "."
        if target != "." and ctx.executor.inside_workdir(target):
            target = ctx.executor.rel(target)
        q = shlex.quote
        width = ctx.cfg.max_line_chars
        if ctx.executor.has_command("rg"):
            parts = [
                "rg",
                "-n",
                "--no-heading",
                "--color=never",
                "--sort=path",
                "--hidden",
                f"--max-columns={width}",
                "--max-columns-preview",
                "-g",
                q("!.git"),
            ]
            if args.get("ignore_case"):
                parts.append("-i")
            if args.get("glob"):
                parts += ["-g", q(args["glob"])]
            parts += ["-e", q(pattern), "--", q(target)]
        else:
            parts = ["grep", "-rnIE", "--color=never", "--exclude-dir=.git"]
            if args.get("ignore_case"):
                parts.append("-i")
            if args.get("glob"):
                parts.append(f"--include={q(args['glob'])}")
            parts += ["-e", q(pattern), "--", q(target)]
        return " ".join(parts)

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if not args["pattern"]:
            raise ToolError("empty pattern")
        r = ctx.executor.run(self.command(ctx, args), timeout=60, merge_stderr=False)
        if r.timed_out:
            raise ToolError("search timed out; narrow the path or glob")
        if r.exit_code == 1 and not r.output.strip():
            return ToolResult("No matches.", meta={"count": 0})
        if r.exit_code not in (0, 1):
            raise ToolError(
                (r.stderr or r.output).strip()[:500] or f"search failed ({r.exit_code})"
            )
        lines = [ln for ln in r.output.splitlines() if ln]
        width = ctx.cfg.max_line_chars
        lines = [ln if len(ln) <= width + 40 else ln[: width + 40] + "…" for ln in lines]
        for i, ln in enumerate(lines):
            if ln.startswith("./"):
                lines[i] = ln[2:]
        limit = ctx.cfg.grep_max_matches
        body = "\n".join(lines[:limit])
        truncated = len(lines) > limit or r.omitted_bytes > 0
        if truncated:
            body += f"\n[{max(len(lines) - limit, 0)}+ more matches; narrow the search]"
        return ToolResult(body, truncated=truncated, meta={"count": len(lines)})
