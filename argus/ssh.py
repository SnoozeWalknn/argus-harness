"""Run tools on a remote host (e.g. a VM) over SSH.

Every operation is one ``ssh host 'script'`` call. A shared control socket
(``ControlMaster``/``ControlPersist``) keeps per-call latency to a few ms
after the first connection. Remote commands are wrapped in coreutils
``timeout`` so they die on the remote side too when a call times out.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import tempfile

from argus.executors import (
    _EACCES,
    _EISDIR,
    _ENOENT,
    CmdResult,
    Executor,
    _run_process,
)

SSH_FAILURE = re.compile(
    r"^ssh: |Connection refused|Could not resolve hostname|Permission denied \(|"
    r"Connection timed out|Host key verification failed|Connection closed by|No route to host",
    re.M,
)


class ExecutorError(RuntimeError):
    """The executor itself failed (e.g. the SSH connection), not the command."""


class SSHExecutor(Executor):
    kind = "ssh"

    def __init__(
        self,
        host: str,
        workdir: str,
        *,
        ssh_command: list[str] | None = None,
        ssh_options: list[str] | None = None,
        control_persist: str = "10m",
        **kw,
    ):
        super().__init__(workdir, **kw)
        self.host = host
        sock_dir = os.path.join(tempfile.gettempdir(), f"argus-ssh-{os.getuid()}")
        os.makedirs(sock_dir, mode=0o700, exist_ok=True)
        self.control_path = os.path.join(sock_dir, "%C")
        self.ssh = list(ssh_command or ["ssh"])
        self.options = [
            "-o", "BatchMode=yes",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_path}",
            "-o", f"ControlPersist={control_persist}",
            *(ssh_options or []),
        ]  # fmt: skip
        self._home: str | None = None
        self._has_timeout: bool | None = None

    def describe(self) -> str:
        return f"ssh:{self.host}:{self.workdir}"

    def process_argv(self, argv: list[str]) -> tuple[list[str], str | None]:
        remote = f"cd {shlex.quote(self.workdir)} && exec {shlex.join(argv)}"
        return [*self.ssh, *self.options, self.host, remote], None

    # -- transport ---------------------------------------------------------------------------------

    def _ssh(
        self,
        script: str,
        *,
        stdin: bytes | None = None,
        timeout: float | None = None,
        merge_stderr: bool = True,
        capture_bytes: int | None = None,
    ):
        argv = [*self.ssh, *self.options, self.host, script]
        out, err, code, timed_out, ms = _run_process(
            argv,
            cwd=None,
            env=None,
            timeout=timeout,
            stdin=stdin,
            merge_stderr=merge_stderr,
            capture_bytes=capture_bytes or self.capture_bytes,
        )
        if code == 255:
            text = (err.text() if err else "") + out.text()
            if SSH_FAILURE.search(text):
                raise ExecutorError(f"ssh to {self.host} failed: {text.strip()[:300]}")
        return out, err, code, timed_out, ms

    def _probe(self) -> None:
        if self._home is not None:
            return
        out, _, code, _, _ = self._ssh(
            'printf "%s\\n" "$HOME"; command -v timeout >/dev/null 2>&1 && echo T || echo N',
            timeout=60,
            merge_stderr=False,
        )
        lines = out.text().splitlines()
        if code != 0 or len(lines) < 2:
            raise ExecutorError(f"cannot probe {self.host}: {out.text()[:200]!r}")
        self._home = lines[0].strip()
        self._has_timeout = lines[1].strip() == "T"

    def home(self) -> str:
        self._probe()
        return self._home or "~"

    # -- commands ----------------------------------------------------------------------------------

    def script(self, cmd: str, cwd: str | None, timeout: float | None) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        shell_cmd = f"{shlex.quote(self.shell)} -c {shlex.quote(cmd)}"
        if timeout and self._has_timeout:
            shell_cmd = f"timeout -k 5 {math.ceil(timeout)} {shell_cmd}"
        return f"cd {shlex.quote(cwd or self.workdir)} && exec env {env} {shell_cmd}"

    def run(
        self,
        cmd: str,
        *,
        timeout: float | None = None,
        stdin: bytes | None = None,
        merge_stderr: bool = True,
        cwd: str | None = None,
        capture_bytes: int | None = None,
        sandbox: object = None,
    ) -> CmdResult:
        if sandbox is not None and getattr(sandbox, "mode", "off") != "off":
            raise ValueError(
                "the SSH executor cannot sandbox commands; the remote host is the boundary"
            )
        self._probe()
        script = self.script(cmd, self.resolve(cwd) if cwd else None, timeout)
        # Local deadline is a backstop; the remote `timeout` normally fires first.
        local_timeout = timeout + 15 if timeout else None
        out, err, code, timed_out, ms = self._ssh(
            script,
            stdin=stdin,
            timeout=local_timeout,
            merge_stderr=merge_stderr,
            capture_bytes=capture_bytes,
        )
        if timeout and code in (124, 137) and ms >= timeout * 1000 * 0.95:
            timed_out, code = True, None
        return CmdResult(
            exit_code=code,
            output=out.text(),
            stderr=err.text() if err else "",
            timed_out=timed_out,
            duration_ms=ms,
            total_bytes=out.total,
            omitted_bytes=max(out.omitted, 0),
        )

    # -- files -------------------------------------------------------------------------------------

    def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes:
        p = shlex.quote(self.resolve(path))
        reader = f"head -c {int(max_bytes)} -- {p}" if max_bytes else f"cat -- {p}"
        script = (
            f"if [ -d {p} ]; then exit {_EISDIR}; elif [ ! -e {p} ]; then exit {_ENOENT}; "
            f"elif [ ! -r {p} ]; then exit {_EACCES}; fi; {reader}"
        )
        cap = (max_bytes or 1 << 30) + 1
        out, err, code, _, _ = self._ssh(script, timeout=120, merge_stderr=False, capture_bytes=cap)
        if code == _ENOENT:
            raise FileNotFoundError(path)
        if code == _EISDIR:
            raise IsADirectoryError(path)
        if code == _EACCES:
            raise PermissionError(path)
        if code != 0:
            raise ExecutorError(f"reading {path} failed: {err.text() if err else ''}")
        return out.raw()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = shlex.quote(self.resolve(path))
        script = f'mkdir -p -- "$(dirname -- {p})" && cat > {p}'
        _, err, code, _, _ = self._ssh(script, stdin=data, timeout=120, merge_stderr=False)
        if code != 0:
            raise ExecutorError(f"writing {path} failed: {err.text() if err else ''}")

    def exists(self, path: str) -> bool:
        _, _, code, _, _ = self._ssh(f"test -e {shlex.quote(self.resolve(path))}", timeout=60)
        return code == 0

    def close(self) -> None:
        """Stop the control master, if one was started."""
        try:
            subprocess.run(
                [*self.ssh, *self.options, "-O", "exit", self.host],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
