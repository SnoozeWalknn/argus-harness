"""Background commands: dev servers, watchers, long builds.

``bash(cmd, background=true)`` starts one and returns at once with its first output;
the ``job`` tool lists them, shows new output (optionally waiting for more), and
stops them. Locally a job runs under the same sandbox as ``bash`` (bubblewrap's
``--die-with-parent`` or Landlock's inherited restrictions), in its own process
group; over SSH it is started with ``setsid nohup``. Jobs end with argus: the TUI
keeps them across model switches, ``argus run`` stops them when the run ends.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj
from argus.tools.truncate import smart_truncate

POLL = 0.1


class Handle:
    """A started background process."""

    pid: int = 0

    def poll(self) -> int | None:  # exit code, None while running
        raise NotImplementedError

    def read(self, offset: int) -> bytes:
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError


class LocalHandle(Handle):
    def __init__(self, proc: subprocess.Popen, log: str):
        self.proc, self.log, self.pid = proc, log, proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()

    def read(self, offset: int) -> bytes:
        try:
            with open(self.log, "rb") as f:
                f.seek(offset)
                return f.read()
        except OSError:
            return b""

    def kill(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                self.proc.wait(timeout=2.0)
                return
            except subprocess.TimeoutExpired:
                continue


class ShellHandle(Handle):
    """A job started through the executor's shell (SSH): polled with shell commands."""

    def __init__(self, executor: Any, pid: int, log: str):
        self.ex, self.pid, self.log = executor, pid, log

    def poll(self) -> int | None:
        r = self.ex.run(
            f"if kill -0 {self.pid} 2>/dev/null; then echo running; "
            f"else cat {shlex.quote(self.log)}.exit 2>/dev/null || echo 255; fi",
            timeout=30,
        )
        out = r.output.strip()
        if out == "running":
            return None
        try:
            return int(out.splitlines()[-1])
        except (ValueError, IndexError):
            return 255

    def read(self, offset: int) -> bytes:
        r = self.ex.run(f"tail -c +{offset + 1} {shlex.quote(self.log)} 2>/dev/null", timeout=30)
        return r.output.encode()

    def kill(self) -> None:
        self.ex.run(
            f"kill -TERM -- -{self.pid} 2>/dev/null; sleep 1; kill -KILL -- -{self.pid} 2>/dev/null",
            timeout=30,
        )


def spawn(executor: Any, cmd: str, sandbox: Any, log_dir: str) -> Handle:
    """Start ``cmd`` in the background on ``executor``, output (merged) going to a log."""
    name = f"{uuid.uuid4().hex[:8]}.log"
    if executor.kind == "local":
        log = os.path.join(log_dir, name)
        env = {**os.environ, **executor.env}
        argv = [executor.shell, "-c", cmd]
        wrapped = None
        backend = getattr(executor, "sandbox_backend", None)
        if sandbox is not None and sandbox.mode != "off" and backend is not None:
            wrapped = backend.wrap(argv, sandbox)
            argv = wrapped.argv
        with open(log, "wb") as out:
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=executor.workdir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=wrapped.preexec_fn if wrapped else None,
                    pass_fds=wrapped.pass_fds if wrapped else (),
                )
            finally:
                if wrapped and wrapped.cleanup:
                    wrapped.cleanup()  # the child has its own copy of anything it needed
        return LocalHandle(proc, log)
    if sandbox is not None and sandbox.mode != "off":
        raise ToolError(f"background jobs cannot be sandboxed on {executor.kind}")
    log = f"{log_dir}/{name}"
    inner = f"{cmd}\necho $? > {shlex.quote(log)}.exit"
    r = executor.run(
        f"mkdir -p {shlex.quote(log_dir)} && "
        f"(setsid nohup {executor.shell} -c {shlex.quote(inner)} > {shlex.quote(log)} 2>&1 "
        f"< /dev/null & echo $!)",
        timeout=30,
    )
    try:
        pid = int(r.output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ToolError(f"could not start the job: {r.output.strip()[:300]}") from None
    return ShellHandle(executor, pid, log)


@dataclass
class Job:
    id: int
    cmd: str
    handle: Handle
    started: float = field(default_factory=time.time)
    offset: int = 0  # bytes of output already shown
    exit_code: int | None = None
    killed: bool = False

    def status(self) -> str:
        if self.killed:
            return "killed"
        return "running" if self.exit_code is None else f"exited {self.exit_code}"


class Jobs:
    """The background jobs of an agent (or of a TUI session, across agents)."""

    def __init__(self, limit: int = 8):
        self.limit = limit
        self.jobs: dict[int, Job] = {}
        self._next = 1
        self._dirs: dict[str, str] = {}  # executor kind:workdir -> log dir

    def _log_dir(self, executor: Any) -> str:
        key = f"{executor.kind}:{executor.workdir}"
        if key not in self._dirs:
            if executor.kind == "local":
                self._dirs[key] = tempfile.mkdtemp(prefix="argus-jobs-")
            else:
                self._dirs[key] = f"/tmp/argus-jobs-{uuid.uuid4().hex[:10]}"
        return self._dirs[key]

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if self.refresh(j) is None]

    def refresh(self, job: Job) -> int | None:
        if job.exit_code is None and not job.killed:
            job.exit_code = job.handle.poll()
        return job.exit_code if not job.killed else -1

    def start(self, executor: Any, cmd: str, sandbox: Any) -> Job:
        if len(self.running()) >= self.limit:
            raise ToolError(
                f'{self.limit} background jobs are already running; stop one with job(action="kill")'
            )
        handle = spawn(executor, cmd, sandbox, self._log_dir(executor))
        job = Job(self._next, cmd, handle)
        self.jobs[job.id] = job
        self._next += 1
        return job

    def get(self, job_id: Any) -> Job:
        try:
            return self.jobs[int(job_id)]
        except (KeyError, TypeError, ValueError):
            known = ", ".join(str(i) for i in self.jobs) or "none"
            raise ToolError(f"no job {job_id!r}; jobs: {known}") from None

    def output(self, job: Job, wait: float = 0.0, settle: float = 0.0) -> str:
        """New output since the last call. Waits up to ``wait`` seconds for the job to
        exit; with ``settle``, returns once output has been quiet that long."""
        end = time.monotonic() + wait
        last_size, quiet_since = -1, time.monotonic()
        while self.refresh(job) is None and time.monotonic() < end:
            if settle:
                size = len(job.handle.read(job.offset))
                if size != last_size:
                    last_size, quiet_since = size, time.monotonic()
                elif size and time.monotonic() - quiet_since >= settle:
                    break
            time.sleep(POLL)
        data = job.handle.read(job.offset)
        job.offset += len(data)
        return data.decode("utf-8", "replace")

    def kill(self, job: Job) -> None:
        if self.refresh(job) is None:
            job.handle.kill()
            job.killed = True

    def close(self) -> None:
        for job in self.jobs.values():
            try:
                self.kill(job)
            except Exception:  # the executor may already be gone
                pass

    def summary(self) -> list[str]:
        return [f"{j.id}  {j.status():<9} {j.cmd[:60]}" for j in self.jobs.values()]


def shown(ctx: ToolContext, text: str) -> tuple[str, bool]:
    c = ctx.cfg
    return smart_truncate(
        text, c.bash_max_chars, c.bash_head_lines, c.bash_tail_lines, c.max_line_chars
    )


class JobTool(Tool):
    name = "job"
    description = (
        "Background commands started with bash(background=true). action=list shows them; "
        "action=output shows a job's new output, waiting up to `wait` seconds for it to "
        "finish; action=kill stops it."
    )
    parameters = obj(
        {
            "action": {"type": "string", "enum": ["list", "output", "kill"]},
            "id": {"type": "integer"},
            "wait": {"type": "integer", "description": "seconds to wait (output; max 60)"},
        },
        ["action"],
    )
    summary = "list background jobs, show a job's new output, or kill it"

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        jobs = ctx.jobs
        if jobs is None:
            raise ToolError("background jobs are not available here")
        action = args.get("action") or "list"
        if action == "list" or args.get("id") is None:
            lines = jobs.summary()
            return ToolResult("\n".join(lines) if lines else "No background jobs.")
        job = jobs.get(args["id"])
        if action == "kill":
            jobs.kill(job)
            text, truncated = shown(ctx, jobs.output(job))
            body = f"{text.rstrip()}\n" if text.strip() else ""
            return ToolResult(f"{body}[job {job.id} {job.status()}]", truncated=truncated)
        wait = min(max(float(args.get("wait") or 0), 0.0), 60.0)
        text, truncated = shown(ctx, jobs.output(job, wait=wait))
        body = text.rstrip() if text.strip() else "(no new output)"
        return ToolResult(
            f"{body}\n[job {job.id} {job.status()}]",
            truncated=truncated,
            meta={"job": job.id, "status": job.status()},
        )
