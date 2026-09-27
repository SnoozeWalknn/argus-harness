from __future__ import annotations

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj
from argus.tools.truncate import smart_truncate


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command in the workspace root (fresh shell each call) and return "
        "its combined stdout/stderr and exit code. background=true starts it as a job "
        "(servers, watchers) and returns its first output; see the job tool."
    )
    parameters = obj(
        {
            "cmd": {"type": "string"},
            "timeout": {"type": "integer", "description": "seconds"},
            "background": {"type": "boolean", "description": "run as a background job"},
        },
        ["cmd"],
    )
    summary = "run a shell command in the workspace (background=true: as a job)"
    mutating = True

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        cmd = args["cmd"]
        if not cmd.strip():
            raise ToolError("empty command")
        if args.get("background"):
            return self.background(ctx, cmd)
        timeout = float(args.get("timeout") or ctx.cfg.bash_timeout)
        timeout = min(max(timeout, 1.0), max(ctx.cfg.bash_timeout * 5, ctx.cfg.bash_timeout))
        r = ctx.executor.run(cmd, timeout=timeout, sandbox=ctx.sandbox)
        text, truncated = self.shrink(ctx, r.output)
        if truncated and r.total_bytes > len(text):
            n_lines = r.output.count("\n") + 1
            text += f"\n[output was {r.total_bytes} bytes / ~{n_lines} lines; filter it with grep, head or tail to see more]"
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
                "sandbox": ctx.sandbox.mode if ctx.sandbox is not None else "off",
            },
        )

    def background(self, ctx: ToolContext, cmd: str) -> ToolResult:
        from argus.tools.jobs import shown

        if ctx.jobs is None:
            raise ToolError("background jobs are not available here; run it without background")
        job = ctx.jobs.start(ctx.executor, cmd, ctx.sandbox)
        # first output: until the job exits, goes quiet for a moment, or a few seconds pass
        text, truncated = shown(ctx, ctx.jobs.output(job, wait=ctx.cfg.job_start_wait, settle=0.5))
        body = f"{text.rstrip()}\n" if text.strip() else ""
        if job.exit_code is not None:
            tail = f"[job {job.id} exited {job.exit_code}]"
        else:
            tail = (
                f'[job {job.id} running (pid {job.handle.pid}); job(action="output", id={job.id}) '
                f'shows new output, job(action="kill", id={job.id}) stops it]'
            )
        return ToolResult(
            f"started job {job.id}: {cmd}\n{body}{tail}",
            truncated=truncated,
            meta={
                "job": job.id,
                "pid": job.handle.pid,
                "sandbox": ctx.sandbox.mode if ctx.sandbox is not None else "off",
            },
        )

    def shrink(self, ctx: ToolContext, output: str) -> tuple[str, bool]:
        c = ctx.cfg
        return smart_truncate(
            output, c.bash_max_chars, c.bash_head_lines, c.bash_tail_lines, c.max_line_chars
        )
