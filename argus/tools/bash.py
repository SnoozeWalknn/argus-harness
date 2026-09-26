from __future__ import annotations

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj
from argus.tools.truncate import head_tail, strip_ansi


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command in the workspace root (fresh shell each call) and return "
        "its combined stdout/stderr and exit code."
    )
    parameters = obj(
        {
            "cmd": {"type": "string"},
            "timeout": {"type": "integer", "description": "seconds"},
        },
        ["cmd"],
    )
    summary = "run a shell command in the workspace"
    mutating = True

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        cmd = args["cmd"]
        if not cmd.strip():
            raise ToolError("empty command")
        timeout = float(args.get("timeout") or ctx.cfg.bash_timeout)
        timeout = min(max(timeout, 1.0), max(ctx.cfg.bash_timeout * 5, ctx.cfg.bash_timeout))
        r = ctx.executor.run(cmd, timeout=timeout)
        text, truncated = self.shrink(ctx, r.output)
        if r.timed_out:
            status = f"[timed out after {timeout:g}s]"
        elif not text.strip():
            status = f"[exit {r.exit_code}, no output]"
        else:
            status = f"[exit {r.exit_code}]"
        body = f"{text.rstrip()}\n{status}" if text.strip() else status
        return ToolResult(
            body,
            ok=True,  # a failing command is a normal result, not a tool error
            truncated=truncated or r.omitted_bytes > 0,
            meta={
                "exit_code": r.exit_code,
                "timed_out": r.timed_out,
                "output_bytes": r.total_bytes,
                "cmd_ms": round(r.duration_ms, 1),
            },
        )

    def shrink(self, ctx: ToolContext, output: str) -> tuple[str, bool]:
        c = ctx.cfg
        return head_tail(strip_ansi(output), c.bash_max_chars, c.bash_head_lines, c.bash_tail_lines)
