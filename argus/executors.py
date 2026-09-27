"""Where commands run and files live: locally or on an SSH host.

Tools only talk to an :class:`Executor`, so they behave identically in both
places. Anything beyond plain file I/O is expressed as a shell command, which
both executors run the same way.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import IO, Any

DEFAULT_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "DEBIAN_FRONTEND": "noninteractive",
}

# Exit codes used by the file-access shell snippets.
_ENOENT, _EISDIR, _EACCES = 20, 21, 22

LIST_FILES_SCRIPT = (
    "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
    "git ls-files -z -co --exclude-standard; printf '\\0::deleted::\\0'; git ls-files -z -d; "
    "elif command -v rg >/dev/null 2>&1; then rg --files --hidden --null -g '!.git'; "
    "else find . -type f -not -path './.git/*' -print0; fi"
)


@dataclass
class CmdResult:
    exit_code: int | None  # None when killed by timeout
    output: str  # stdout (+stderr when merged), middle elided if over the capture limit
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0
    total_bytes: int = 0
    omitted_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class _Sink:
    """Keeps the first ``head`` and last ``tail`` bytes of a stream."""

    def __init__(self, head: int, tail: int):
        self.head_limit, self.tail_limit = head, tail
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.head_limit - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > self.tail_limit:
                del self.tail[: len(self.tail) - self.tail_limit]

    @property
    def omitted(self) -> int:
        return self.total - len(self.head) - len(self.tail)

    def raw(self) -> bytes:
        return bytes(self.head) + bytes(self.tail)

    def text(self) -> str:
        head = self.head.decode("utf-8", "replace")
        if self.omitted <= 0:
            return (bytes(self.head) + bytes(self.tail)).decode("utf-8", "replace")
        tail = self.tail.decode("utf-8", "replace")
        return f"{head}\n[... {self.omitted} bytes omitted ...]\n{tail}"


def _pump(stream: IO[bytes], sink: _Sink) -> None:
    try:
        while True:
            chunk = stream.read1(65536)  # type: ignore[attr-defined]
            if not chunk:
                break
            sink.feed(chunk)
    except (OSError, ValueError):
        pass


def _run_process(
    argv: list[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float | None,
    stdin: bytes | None,
    merge_stderr: bool,
    capture_bytes: int,
    preexec_fn: Any = None,
    pass_fds: tuple[int, ...] = (),
) -> tuple[_Sink, _Sink | None, int | None, bool, float]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        start_new_session=True,  # own process group, so a timeout kills children too
        preexec_fn=preexec_fn,
        pass_fds=pass_fds,
    )
    out = _Sink(capture_bytes, capture_bytes)
    err = None if merge_stderr else _Sink(capture_bytes, capture_bytes)
    threads = [threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True)]
    if err is not None:
        threads.append(threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True))
    if stdin is not None:

        def _feed() -> None:
            try:
                proc.stdin.write(stdin)  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

        threads.append(threading.Thread(target=_feed, daemon=True))
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
    except BaseException:
        _kill_group(proc)
        raise
    for t in threads:
        # A daemonised grandchild can hold the pipe open forever; don't wait on it.
        t.join(timeout=2.0)
    code = None if timed_out else proc.returncode
    return out, err, code, timed_out, (time.perf_counter() - t0) * 1000


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            continue


class Executor(ABC):
    kind = "abstract"
    can_sandbox = False

    def __init__(
        self,
        workdir: str,
        *,
        shell: str = "bash",
        env: dict[str, str] | None = None,
        capture_bytes: int = 256_000,
    ):
        self.workdir = workdir.rstrip("/") or "/"
        self.shell = shell
        self.env = {**DEFAULT_ENV, **(env or {})}
        self.capture_bytes = capture_bytes
        self._commands: dict[str, bool] = {}

    # -- paths -----------------------------------------------------------------------------

    def resolve(self, path: str) -> str:
        """Absolute, normalised path; relative paths are taken from the workdir."""
        path = path.strip()
        if path.startswith("~"):
            path = posixpath.join(self.home(), path[2:] if path.startswith("~/") else "")
        if not posixpath.isabs(path):
            path = posixpath.join(self.workdir, path)
        return posixpath.normpath(path)

    def rel(self, path: str) -> str:
        """Path relative to the workdir when inside it (shorter for the model), else absolute."""
        abs_path = self.resolve(path)
        if abs_path == self.workdir:
            return "."
        if abs_path.startswith(self.workdir.rstrip("/") + "/"):
            return abs_path[len(self.workdir.rstrip("/")) + 1 :]
        return abs_path

    def inside_workdir(self, path: str) -> bool:
        abs_path = self.resolve(path)
        return abs_path == self.workdir or abs_path.startswith(self.workdir.rstrip("/") + "/")

    # -- commands ----------------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        cmd: str,
        *,
        timeout: float | None = None,
        stdin: bytes | None = None,
        merge_stderr: bool = True,
        cwd: str | None = None,
        capture_bytes: int | None = None,
        sandbox: Any = None,
    ) -> CmdResult:
        """Run ``cmd`` with the configured shell in ``cwd`` (default: workdir).

        ``sandbox`` is a :class:`argus.sandbox.Spec`; executors that cannot sandbox
        refuse it (argus only passes one when the executor supports it).
        """

    def has_command(self, name: str) -> bool:
        if name not in self._commands:
            r = self.run(f"command -v {shlex.quote(name)} >/dev/null 2>&1", timeout=15)
            self._commands[name] = r.exit_code == 0
        return self._commands[name]

    def home(self) -> str:
        return os.path.expanduser("~")

    # -- files -------------------------------------------------------------------------------

    @abstractmethod
    def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    def list_files(self, timeout: float = 60) -> list[str]:
        """Files under the workdir (relative paths), honouring .gitignore when possible."""
        r = self.run(
            LIST_FILES_SCRIPT, timeout=timeout, merge_stderr=False, capture_bytes=64_000_000
        )
        raw = r.output
        deleted: set[str] = set()
        if "\0::deleted::\0" in raw:
            raw, dele = raw.split("\0::deleted::\0", 1)
            deleted = {p for p in dele.split("\0") if p}
        files = []
        seen = set()
        for p in raw.split("\0"):
            if not p or p in deleted:
                continue
            if p.startswith("./"):
                p = p[2:]
            if p not in seen:
                seen.add(p)
                files.append(p)
        return sorted(files)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"{self.kind}:{self.workdir}"


class LocalExecutor(Executor):
    kind = "local"
    can_sandbox = True

    def __init__(self, workdir: str | None = None, **kw):
        super().__init__(os.path.abspath(os.path.expanduser(workdir or os.getcwd())), **kw)
        self.sandbox_backend: Any = None  # argus.sandbox.Backend, set by the agent

    def run(
        self,
        cmd: str,
        *,
        timeout: float | None = None,
        stdin: bytes | None = None,
        merge_stderr: bool = True,
        cwd: str | None = None,
        capture_bytes: int | None = None,
        sandbox: Any = None,
    ) -> CmdResult:
        env = {**os.environ, **self.env}
        argv = [self.shell, "-c", cmd]
        wrapped = None
        if sandbox is not None and sandbox.mode != "off" and self.sandbox_backend is not None:
            wrapped = self.sandbox_backend.wrap(argv, sandbox)
            argv = wrapped.argv
        try:
            out, err, code, timed_out, ms = _run_process(
                argv,
                cwd=self.resolve(cwd) if cwd else self.workdir,
                env=env,
                timeout=timeout,
                stdin=stdin,
                merge_stderr=merge_stderr,
                capture_bytes=capture_bytes or self.capture_bytes,
                preexec_fn=wrapped.preexec_fn if wrapped else None,
                pass_fds=wrapped.pass_fds if wrapped else (),
            )
        finally:
            if wrapped and wrapped.cleanup:
                wrapped.cleanup()
        return CmdResult(
            exit_code=code,
            output=out.text(),
            stderr=err.text() if err else "",
            timed_out=timed_out,
            duration_ms=ms,
            total_bytes=out.total,
            omitted_bytes=max(out.omitted, 0),
        )

    def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes:
        with open(self.resolve(path), "rb") as f:
            return f.read(max_bytes) if max_bytes else f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.resolve(path)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        mode = None
        try:
            mode = os.stat(target).st_mode & 0o7777
        except FileNotFoundError:
            pass
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".argus-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            if mode is not None:
                os.chmod(tmp, mode)
            else:
                os.chmod(tmp, 0o666 & ~_umask())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def exists(self, path: str) -> bool:
        return os.path.exists(self.resolve(path))


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def make_executor(cfg, capture: int = 256_000) -> Executor:
    """Build an executor from an :class:`~argus.config.ExecutorConfig`."""
    if cfg.kind == "ssh":
        from argus.ssh import SSHExecutor

        return SSHExecutor(
            cfg.host,
            cfg.workdir,
            ssh_command=cfg.ssh_command,
            ssh_options=cfg.ssh_options,
            control_persist=cfg.control_persist,
            shell=cfg.shell,
            env=cfg.env,
            capture_bytes=capture,
        )
    return LocalExecutor(cfg.workdir or None, shell=cfg.shell, env=cfg.env, capture_bytes=capture)
