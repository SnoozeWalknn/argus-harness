"""Task suites: saved tasks replayed in fresh workspaces, optionally under several configs.

Suite file (TOML)::

    name = "smoke"
    oracle = false                 # run the check after every mutating turn

    [defaults]                     # applied to every task that does not set the key
    check_timeout = 300
    overrides = ["agent.max_turns=30"]

    [[task]]
    id = "fix-add"
    prompt = "Fix add() in calc.py so the tests pass."
    workspace = "fixtures/calc"    # copied into a fresh workspace (relative to this file)
    # repo = "https://…"; ref = "main"   (alternative: clone)
    setup = ["pip install -q -e ."]
    check = "python3 -m pytest -q" # pass = exit 0
    overrides = ["agent.max_turns=20"]
    tags = ["python"]

Workspaces are created through the executor (a temp dir locally or on the SSH
host) and filled by streaming a tar archive, so suites run the same way on
a VM. Variants are interleaved per task in a seeded random order, so drift
(thermal throttling, a warm cache) does not systematically favour one config.
"""

from __future__ import annotations

import copy
import io
import json
import random
import shlex
import tarfile
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from argus.agent import Agent, Reporter, RunResult
from argus.config import Config, ConfigError, apply_override
from argus.executors import Executor, make_executor
from argus.store import Store

TASK_KEYS = {
    "id",
    "prompt",
    "workspace",
    "repo",
    "ref",
    "setup",
    "check",
    "check_timeout",
    "overrides",
    "tags",
}
DEFAULT_KEYS = TASK_KEYS - {"id", "prompt"}
SUITE_KEYS = {"name", "oracle", "defaults", "task"}
CHECK_OUTPUT_CHARS = 4000


class SuiteError(ValueError):
    pass


@dataclass
class Task:
    id: str
    prompt: str
    workspace: str | None = None  # absolute local path of a directory to copy
    repo: str | None = None
    ref: str | None = None
    setup: list[str] = field(default_factory=list)
    check: str | None = None
    check_timeout: float = 300.0
    overrides: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    path: Path
    tasks: list[Task]
    oracle: bool = False

    def select(self, ids: list[str] | None) -> list[Task]:
        if not ids:
            return self.tasks
        known = {t.id for t in self.tasks}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SuiteError(f"unknown task id(s): {', '.join(missing)}")
        return [t for t in self.tasks if t.id in ids]


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text())
    except FileNotFoundError:
        raise SuiteError(f"suite not found: {p}") from None
    except tomllib.TOMLDecodeError as e:
        raise SuiteError(f"{p}: {e}") from None
    unknown = set(data) - SUITE_KEYS
    if unknown:
        raise SuiteError(f"{p}: unknown key(s) {', '.join(sorted(unknown))}")
    defaults = data.get("defaults", {})
    if set(defaults) - DEFAULT_KEYS:
        raise SuiteError(
            f"{p}: unknown [defaults] key(s) {', '.join(sorted(set(defaults) - DEFAULT_KEYS))}"
        )
    tasks, seen = [], set()
    for i, raw in enumerate(data.get("task", [])):
        bad = set(raw) - TASK_KEYS
        if bad:
            raise SuiteError(f"{p}: task {i}: unknown key(s) {', '.join(sorted(bad))}")
        merged = {**defaults, **raw}
        if not merged.get("prompt"):
            raise SuiteError(f"{p}: task {i} has no prompt")
        tid = str(merged.get("id") or f"task{i + 1}")
        if tid in seen:
            raise SuiteError(f"{p}: duplicate task id {tid!r}")
        seen.add(tid)
        ws = merged.get("workspace")
        if ws:
            ws = str((p.parent / ws).resolve())
            if not Path(ws).is_dir():
                raise SuiteError(f"{p}: task {tid}: workspace {ws} is not a directory")
        overrides = list(merged.get("overrides", []))
        try:  # fail at load time, not halfway through a batch
            probe = Config()
            for o in overrides:
                apply_override(probe, o)
        except ConfigError as e:
            raise SuiteError(f"{p}: task {tid}: {e}") from None
        setup = merged.get("setup", [])
        tasks.append(
            Task(
                id=tid,
                prompt=merged["prompt"],
                workspace=ws,
                repo=merged.get("repo"),
                ref=merged.get("ref"),
                setup=[setup] if isinstance(setup, str) else list(setup),
                check=merged.get("check"),
                check_timeout=float(merged.get("check_timeout", 300)),
                overrides=overrides,
                tags=list(merged.get("tags", [])),
            )
        )
    if not tasks:
        raise SuiteError(f"{p}: no [[task]] entries")
    return Suite(data.get("name") or p.stem, p, tasks, bool(data.get("oracle", False)))


def _toml_value(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v) if "\n" not in v else '"""\n' + v.replace('"""', '\\"\\"\\"') + '"""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(v)


def append_task(path: str | Path, task: dict[str, Any]) -> None:
    """Append a [[task]] table to a suite file (created if missing)."""
    p = Path(path)
    lines = ["", "[[task]]"] + [
        f"{k} = {_toml_value(v)}" for k, v in task.items() if v not in (None, [], "")
    ]
    text = p.read_text() if p.exists() else f'name = "{p.stem}"\n'
    p.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    load_suite(p)  # validate what we wrote


# -- workspaces --------------------------------------------------------------------------------


def create_workdir(cfg: Config, task: Task) -> str:
    if cfg.executor.kind == "local":
        return tempfile.mkdtemp(prefix=f"argus-{task.id}-")
    boot_cfg = copy.deepcopy(cfg.executor)
    boot_cfg.workdir = "/tmp"
    boot = make_executor(boot_cfg)
    r = boot.run(
        f"mktemp -d -t {shlex.quote('argus-' + task.id)}-XXXXXX", timeout=60, merge_stderr=False
    )
    if r.exit_code != 0 or not r.output.strip():
        raise SuiteError(f"cannot create a remote workspace: {r.stderr.strip()}")
    return r.output.strip()


def upload_dir(ex: Executor, local_dir: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(local_dir, arcname=".")
    r = ex.run("tar -xf -", stdin=buf.getvalue(), timeout=600, merge_stderr=True)
    if r.exit_code != 0:
        raise SuiteError(f"copying {local_dir} into the workspace failed: {r.output[:300]}")


def prepare_workspace(cfg: Config, task: Task) -> Executor:
    """Create and populate a fresh workspace; returns an executor rooted in it."""
    ex_cfg = copy.deepcopy(cfg.executor)
    ex_cfg.workdir = create_workdir(cfg, task)
    ex = make_executor(ex_cfg, cfg.tools.capture_bytes)
    if task.workspace:
        upload_dir(ex, task.workspace)
    elif task.repo:
        cmd = f"git clone -q {shlex.quote(task.repo)} ."
        if task.ref:
            cmd += f" && git checkout -q {shlex.quote(task.ref)}"
        r = ex.run(cmd, timeout=900)
        if r.exit_code != 0:
            raise SuiteError(f"clone failed: {r.output[:300]}")
    for cmd in task.setup:
        r = ex.run(cmd, timeout=900)
        if r.exit_code != 0:
            raise SuiteError(f"setup command failed ({r.exit_code}): {cmd}\n{r.output[-500:]}")
    return ex


def run_check(ex: Executor, task: Task) -> tuple[bool, str]:
    r = ex.run(task.check or "true", timeout=task.check_timeout)
    out = r.output
    if len(out) > CHECK_OUTPUT_CHARS:
        out = out[: CHECK_OUTPUT_CHARS // 4] + "\n[...]\n" + out[-3 * CHECK_OUTPUT_CHARS // 4 :]
    if r.timed_out:
        out += f"\n[check timed out after {task.check_timeout:g}s]"
    return r.ok, out


# -- running -------------------------------------------------------------------------------------


@dataclass
class Outcome:
    task: str
    variant: str
    rep: int
    result: RunResult | None
    passed: bool
    workdir: str
    error: str | None = None


class Oracle:
    """Runs the task check after each mutating turn to find when the task was really done."""

    def __init__(self, ex: Executor, task: Task):
        self.ex = ex
        self.task = task
        self.first_pass: int | None = None

    def __call__(self, turn: int, run: Any) -> None:
        if not run.turn_mutated:
            return
        passed, _ = run_check(self.ex, self.task)
        run.store.add_event(run.id, turn, "oracle", {"passed": passed})
        if passed and self.first_pass is None:
            self.first_pass = turn


class SuiteRunner:
    def __init__(
        self,
        suite: Suite,
        variants: list[tuple[str, Config]],
        store: Store,
        *,
        repeat: int = 1,
        task_ids: list[str] | None = None,
        oracle: bool | None = None,
        keep: bool = False,
        reporter_factory: Callable[[], Reporter] | None = None,
        progress: Callable[[str], None] | None = None,
        kind: str = "suite",
    ):
        self.suite = suite
        self.variants = variants
        self.store = store
        self.repeat = repeat
        self.tasks = suite.select(task_ids)
        self.oracle = suite.oracle if oracle is None else oracle
        self.keep = keep
        self.reporter_factory = reporter_factory or Reporter
        self.progress = progress or (lambda s: None)
        self.kind = kind
        self.batch_id = ""

    def run(self) -> tuple[str, list[Outcome]]:
        self.batch_id = self.store.new_batch(
            self.kind,
            self.suite.name,
            {
                "suite": str(self.suite.path),
                "variants": {
                    label: {"name": c.name, "hash": c.hash()} for label, c in self.variants
                },
                "tasks": [t.id for t in self.tasks],
                "repeat": self.repeat,
                "oracle": self.oracle,
            },
        )
        outcomes = []
        total = self.repeat * len(self.tasks) * len(self.variants)
        n = 0
        for rep in range(self.repeat):
            for task in self.tasks:
                order = list(self.variants)
                random.Random(f"{self.batch_id}:{rep}:{task.id}").shuffle(order)
                for label, cfg in order:
                    n += 1
                    self.progress(f"[{n}/{total}] {task.id} · {label} · rep {rep + 1}")
                    outcomes.append(self.run_one(task, label, cfg, rep))
        return self.batch_id, outcomes

    def run_one(self, task: Task, label: str, base: Config, rep: int) -> Outcome:
        cfg = copy.deepcopy(base)
        for o in task.overrides:
            apply_override(cfg, o)
        try:
            ex = prepare_workspace(cfg, task)
        except Exception as e:
            self.progress(f"  setup failed: {e}")
            return Outcome(task.id, label, rep, None, False, "", error=f"setup: {e}")
        cfg.executor.workdir = ex.workdir
        agent = Agent(cfg, executor=ex, store=self.store, reporter=self.reporter_factory())
        oracle = Oracle(ex, task) if self.oracle and task.check else None
        try:
            result = agent.run(
                task.prompt,
                task_id=task.id,
                batch_id=self.batch_id,
                variant=label,
                after_turn=oracle,
            )
            self.store.update_run(result.run_id, rep=rep)
            passed = result.status == "completed"
            if task.check:
                passed, output = run_check(ex, task)
                self.store.update_run(result.run_id, check_passed=int(passed), check_output=output)
            if oracle:
                self.judge_overrun(result, oracle, passed, cfg)
            if passed and not self.keep:
                ex.run(f"cd / && rm -rf -- {shlex.quote(ex.workdir)}", timeout=120)
        finally:
            agent.close()
        status = "PASS" if passed else "FAIL"
        self.progress(f"  {status} {result.status} · {result.turns} turns · run {result.run_id}")
        return Outcome(task.id, label, rep, result, passed, ex.workdir)

    def judge_overrun(
        self, result: RunResult, oracle: Oracle, passed_at_end: bool, cfg: Config
    ) -> None:
        if oracle.first_pass is None:
            return
        k = oracle.first_pass
        extra = (result.turns - 1) - (k + 1)  # tool turns beyond the one that should have finished
        details = []
        if extra >= cfg.agent.overrun_turns:
            details.append(
                f"task check passed after turn {k}; the agent continued {extra} more tool turns"
            )
        if not passed_at_end:
            details.append(f"task check passed after turn {k} but failed at the end (regressed)")
        for d in details:
            self.store.add_failure(result.run_id, k, "overrun", d)
            result.failures.append(("overrun", d))
        self.store.add_event(
            result.run_id, k, "oracle_summary", {"first_pass_turn": k, "extra_turns": extra}
        )
