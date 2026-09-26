"""Workspace snapshots and checkpoints in a shadow git repository.

The shadow repository has its own git dir, index and refs, and uses the
workspace as its work tree. It never touches the user's repository (index,
branches, HEAD, stash), honours the workspace's ``.gitignore`` files, and
works in directories that are not git repositories at all. It lives on the
executor's host, so checkpoints work the same over SSH.

A snapshot is ``git add -A && git write-tree``; with a warm index this costs
a few milliseconds. Checkpoints are commits chained on ``refs/argus/<run>``.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import shlex
from dataclasses import dataclass

from argus.config import CheckpointConfig
from argus.executors import Executor

IDENTITY = "-c user.name=argus -c user.email=argus@localhost -c commit.gpgsign=false"


class GitError(RuntimeError):
    pass


@dataclass
class Change:
    status: str  # A | M | D | T
    path: str

    def __str__(self) -> str:
        return f"{self.status} {self.path}"


@dataclass
class Checkpoint:
    commit: str
    tree: str


def summarize(changes: list[Change], limit: int = 8) -> str:
    shown = ", ".join(str(c) for c in changes[:limit])
    if len(changes) > limit:
        shown += f" (+{len(changes) - limit} more)"
    return shown


class ShadowGit:
    def __init__(self, executor: Executor, root: str = "", excludes: list[str] | None = None):
        self.ex = executor
        self.excludes = excludes or []
        key = hashlib.sha1(executor.workdir.encode()).hexdigest()[:16]
        root = root or os.environ.get("ARGUS_SHADOW_ROOT") or "~/.local/share/argus/shadow"
        if root.startswith("~"):
            root = executor.home() + root[1:]
        self.git_dir = posixpath.join(root, f"{key}.git")
        self._env = (
            f"GIT_DIR={shlex.quote(self.git_dir)} GIT_WORK_TREE={shlex.quote(executor.workdir)} "
            f"GIT_INDEX_FILE={shlex.quote(self.git_dir + '/argus-index')} "
            "GIT_OPTIONAL_LOCKS=0"
        )

    def git(self, args: str, timeout: float = 120, stdin: bytes | None = None) -> str:
        cmd = f"{self._env} git {IDENTITY} -c core.autocrlf=false -c core.quotepath=off {args}"
        r = self.ex.run(
            cmd, timeout=timeout, merge_stderr=False, stdin=stdin, capture_bytes=64_000_000
        )
        if r.timed_out:
            raise GitError(f"git {args.split()[0]} timed out after {timeout}s")
        if r.exit_code != 0:
            raise GitError(
                f"git {args.split()[0]} failed ({r.exit_code}): {r.stderr.strip()[:300]}"
            )
        return r.output

    def init(self) -> None:
        gd = shlex.quote(self.git_dir)
        excludes = "\n".join(self.excludes) + "\n"
        script = (
            f"if [ ! -f {gd}/HEAD ]; then mkdir -p {gd} && git init -q --bare {gd} "
            f"&& git --git-dir={gd} config core.bare false; fi && "
            f"mkdir -p {gd}/info && cat > {gd}/info/exclude"
        )
        r = self.ex.run(script, timeout=60, merge_stderr=False, stdin=excludes.encode())
        if r.exit_code != 0:
            raise GitError(
                f"cannot initialise shadow repository {self.git_dir}: {r.stderr.strip()[:300]}"
            )

    def snapshot(self, timeout: float = 120) -> str:
        """Tree id of the current workspace state."""
        out = self.git("add -A . >/dev/null && " + f"{self._env} git write-tree", timeout=timeout)
        return out.strip().splitlines()[-1]

    def commit(self, tree: str, message: str, parent: str | None = None) -> str:
        p = f"-p {parent} " if parent else ""
        return self.git(f"commit-tree {tree} {p}-m {shlex.quote(message)}").strip()

    def update_ref(self, ref: str, commit: str) -> None:
        self.git(f"update-ref {shlex.quote(ref)} {commit}")

    def resolve(self, rev: str) -> str:
        return self.git(f"rev-parse --verify -q {shlex.quote(rev)}").strip()

    def tree_of(self, commit: str) -> str:
        return self.resolve(f"{commit}^{{tree}}")

    def diff(self, a: str, b: str) -> list[Change]:
        out = self.git(f"diff-tree -r --no-renames --name-status -z {a} {b}")
        parts = [p for p in out.split("\0") if p]
        return [Change(parts[i][0], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]

    def patch(self, a: str, b: str, paths: list[str] | None = None) -> str:
        spec = " -- " + " ".join(shlex.quote(p) for p in paths) if paths else ""
        return self.git(f"diff --no-color {a} {b}{spec}")

    def is_ignored(self, rel_path: str) -> bool:
        cmd = f"{self._env} git check-ignore -q -- {shlex.quote(rel_path)}"
        return self.ex.run(cmd, timeout=30).exit_code == 0

    def restore(self, target_commit: str) -> list[Change]:
        """Make the workspace match ``target_commit``; returns what was changed."""
        current = self.snapshot()
        target_tree = self.tree_of(target_commit)
        changes = self.diff(current, target_tree)  # how to get from now to target
        if not changes:
            return []
        added = [c.path for c in changes if c.status == "D"]  # present now, absent in target
        if added:
            data = "\0".join(added).encode() + b"\0"
            r = self.ex.run("xargs -0 rm -f --", stdin=data, timeout=120, merge_stderr=False)
            if r.exit_code != 0:
                raise GitError(f"removing files failed: {r.stderr[:300]}")
        if any(c.status != "D" for c in changes):
            self.git(f"checkout {target_commit} -- .")
        return changes


class Checkpointer:
    """Checkpoints for one run: a baseline, one before each mutating call, and a final one."""

    def __init__(self, executor: Executor, cfg: CheckpointConfig, run_id: str):
        self.executor = executor
        self.cfg = cfg
        self.git: ShadowGit | None = None
        self.ref = f"refs/argus/{run_id}"
        self.last: Checkpoint | None = None
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self.error is None and self.last is not None

    def start(self) -> Checkpoint | None:
        try:
            if not self.executor.has_command("git"):
                raise GitError("git is not installed on the executor host")
            self.git = ShadowGit(self.executor, self.cfg.shadow_root, self.cfg.excludes)
            self.git.init()
            return self.checkpoint("baseline")
        except Exception as e:  # checkpoints are a safety net, never a reason to fail a run
            self.error = f"{type(e).__name__}: {e}"
            return None

    def checkpoint(self, message: str) -> Checkpoint:
        tree = self.git.snapshot()
        if self.last and tree == self.last.tree:
            return self.last  # nothing changed since the previous checkpoint
        commit = self.git.commit(tree, message, self.last.commit if self.last else None)
        self.git.update_ref(self.ref, commit)
        self.last = Checkpoint(commit, tree)
        return self.last

    def changes_since(self, cp: Checkpoint) -> tuple[str, list[Change]]:
        tree = self.git.snapshot()
        return tree, self.git.diff(cp.tree, tree) if tree != cp.tree else []


def check_edit(changes: list[Change], rel_path: str) -> str | None:
    """The only change an edit may cause is to its own file."""
    paths = {c.path for c in changes}
    if rel_path not in paths:
        return f"edit reported success but {rel_path} is unchanged on disk"
    others = [c for c in changes if c.path != rel_path]
    if others:
        return f"edit of {rel_path} also changed: {summarize(others)}"
    return None


def mass_deletion(changes: list[Change], threshold: int = 20) -> str | None:
    deleted = [c for c in changes if c.status == "D"]
    if len(deleted) >= threshold:
        return f"command deleted {len(deleted)} files: {summarize(deleted, 5)}"
    return None
