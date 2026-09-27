"""Sandboxed command execution on Linux: bubblewrap or Landlock.

Two modes, as in Codex:

``read-only``        the command can read everything and write nothing (a private
                     ``/tmp`` under bubblewrap); no network
``workspace-write``  writes are allowed only in the workspace, ``/tmp`` and
                     ``sandbox.writable`` paths; network per ``sandbox.network``

Backends:

``bwrap``     bubblewrap: the root filesystem bound read-only, writable paths bound
              read-write on top, ``--unshare-net`` without network. Needs
              unprivileged user namespaces, which some distributions restrict.
``landlock``  the kernel's Landlock LSM, applied in the child just before ``exec``
              (raw syscalls through ``ctypes``, no helper binary). The ruleset is
              built in the parent; the child only calls ``prctl(NO_NEW_PRIVS)`` and
              ``landlock_restrict_self``. ABI ≥ 4 can also deny TCP.
``none``      nothing; commands run unsandboxed.

``auto`` picks the first backend that actually works here. The sandbox wraps the
model's shell commands only; argus's own git checkpoints and task checks are not
sandboxed.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

READ_ONLY = "read-only"
WORKSPACE_WRITE = "workspace-write"
OFF = "off"
MODES = (READ_ONLY, WORKSPACE_WRITE, OFF)
BACKENDS = ("auto", "bwrap", "landlock", "none")

# Landlock (linux/landlock.h). The syscall numbers are the same on every architecture.
SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13  # ABI 2
FS_TRUNCATE = 1 << 14  # ABI 3
FS_IOCTL_DEV = 1 << 15  # ABI 5
NET_BIND_TCP = 1 << 0  # ABI 4
NET_CONNECT_TCP = 1 << 1


class SandboxError(RuntimeError):
    pass


@dataclass
class Spec:
    """How to confine one command."""

    mode: str = WORKSPACE_WRITE
    workdir: str = ""
    network: bool = True
    writable: list[str] = field(default_factory=list)

    def writable_roots(self) -> list[str]:
        if self.mode != WORKSPACE_WRITE:
            return []
        roots = [self.workdir, "/tmp", os.environ.get("TMPDIR", "")]
        roots += [os.path.expanduser(p) for p in self.writable]
        out = []
        for r in roots:
            if r and os.path.isdir(r) and os.path.realpath(r) not in out:
                out.append(os.path.realpath(r))
        return out

    @property
    def net(self) -> bool:
        return self.network and self.mode == WORKSPACE_WRITE


@dataclass
class Wrapped:
    argv: list[str]
    preexec_fn: Callable[[], None] | None = None
    pass_fds: tuple[int, ...] = ()
    cleanup: Callable[[], None] | None = None


class Backend:
    name = "none"

    def wrap(self, argv: list[str], spec: Spec) -> Wrapped:
        return Wrapped(argv)

    def limits(self, spec: Spec) -> list[str]:
        """What this backend cannot enforce for ``spec`` (reported to the user)."""
        return []


class NoSandbox(Backend):
    pass


class Bwrap(Backend):
    name = "bwrap"

    def __init__(self, exe: str = "bwrap"):
        self.exe = exe

    def wrap(self, argv: list[str], spec: Spec) -> Wrapped:
        args = [self.exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        if spec.mode == WORKSPACE_WRITE:
            for root in spec.writable_roots():
                args += ["--bind", root, root]
        else:
            args += ["--tmpfs", "/tmp"]  # scratch space that vanishes with the command
            if spec.workdir:
                args += ["--ro-bind", spec.workdir, spec.workdir]  # visible again if under /tmp
        if not spec.net:
            args.append("--unshare-net")
        if spec.workdir:
            args += ["--chdir", spec.workdir]
        args += ["--die-with-parent", "--new-session", "--"]
        return Wrapped(args + argv)


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


def landlock_abi() -> int:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        abi = libc.syscall(SYS_CREATE_RULESET, None, ctypes.c_size_t(0), CREATE_RULESET_VERSION)
    except (OSError, AttributeError):
        return 0
    return max(int(abi), 0)


class Landlock(Backend):
    name = "landlock"

    def __init__(self) -> None:
        self.abi = landlock_abi()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self._restrict = self.libc.syscall
        self._prctl = self.libc.prctl

    def _fs_rights(self) -> int:
        rights = (1 << 13) - 1  # EXECUTE .. MAKE_SYM
        if self.abi >= 2:
            rights |= FS_REFER
        if self.abi >= 3:
            rights |= FS_TRUNCATE
        if self.abi >= 5:
            rights |= FS_IOCTL_DEV
        return rights

    def _syscall(self, *args: object) -> int:
        n = self.libc.syscall(*args)
        if n < 0:
            err = ctypes.get_errno()
            raise SandboxError(f"landlock syscall {args[0]} failed: {os.strerror(err)}")
        return n

    def _allow(self, ruleset: int, path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            rule = _PathBeneath(access, fd)
            self._syscall(SYS_ADD_RULE, ruleset, RULE_PATH_BENEATH, ctypes.byref(rule), 0)
        finally:
            os.close(fd)

    def ruleset(self, spec: Spec) -> int:
        handled = self._fs_rights()
        attr = _RulesetAttr(handled, 0)
        size = 8
        if self.abi >= 4 and not spec.net:
            attr.handled_access_net = NET_BIND_TCP | NET_CONNECT_TCP  # no rules: all TCP denied
            size = 16
        fd = self._syscall(SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(size), 0)
        read = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
        self._allow(fd, "/", read)
        dev = read | FS_WRITE_FILE | (FS_IOCTL_DEV if self.abi >= 5 else 0)
        self._allow(fd, "/dev", dev)  # /dev/null, /dev/tty
        for root in spec.writable_roots():
            self._allow(fd, root, handled)
        if spec.mode == WORKSPACE_WRITE and os.path.isdir("/dev/shm"):
            self._allow(fd, "/dev/shm", handled)
        return fd

    def wrap(self, argv: list[str], spec: Spec) -> Wrapped:
        fd = self.ruleset(spec)
        syscall, prctl = self._restrict, self._prctl

        def restrict() -> None:  # runs in the child: nothing but two syscalls
            if prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
                raise OSError("prctl(PR_SET_NO_NEW_PRIVS) failed")
            if syscall(SYS_RESTRICT_SELF, fd, 0) != 0:
                raise OSError("landlock_restrict_self failed")

        return Wrapped(argv, restrict, (fd,), lambda: os.close(fd))

    def limits(self, spec: Spec) -> list[str]:
        if not spec.net and self.abi < 4:
            return [f"Landlock ABI {self.abi} cannot block the network (needs ABI 4)"]
        return []


@cache
def bwrap_works(exe: str = "bwrap") -> bool:
    path = shutil.which(exe)
    if not path:
        return False
    probe = [path, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net"]
    try:
        r = subprocess.run([*probe, "--", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@cache
def landlock_works() -> bool:
    if landlock_abi() < 1:
        return False
    try:
        backend = Landlock()
        w = backend.wrap(["true"], Spec(READ_ONLY, "/", False))
        r = subprocess.run(
            w.argv, preexec_fn=w.preexec_fn, pass_fds=w.pass_fds, capture_output=True, timeout=10
        )
        if w.cleanup:
            w.cleanup()
    except (OSError, SandboxError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def choose(backend: str = "auto") -> Backend:
    """The requested backend, or for ``auto`` the first one that works on this machine."""
    if backend not in BACKENDS:
        raise SandboxError(f"unknown sandbox backend {backend!r}; choose from {BACKENDS}")
    if backend in ("auto", "bwrap") and bwrap_works():
        return Bwrap()
    if backend in ("auto", "landlock") and landlock_works():
        return Landlock()
    if backend in ("bwrap", "landlock"):
        raise SandboxError(f"the {backend} sandbox is not available on this machine")
    return NoSandbox()


def status() -> dict[str, object]:
    """For ``argus doctor``: what is available here."""
    return {
        "bwrap": bool(shutil.which("bwrap")),
        "bwrap_works": bwrap_works(),
        "landlock_abi": landlock_abi(),
        "landlock_works": landlock_works(),
        "auto": choose("auto").name,
    }
