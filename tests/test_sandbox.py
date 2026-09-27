"""Sandboxed commands (bubblewrap, Landlock) and approval policies."""

from __future__ import annotations

import socket
import subprocess
import threading

import pytest

from argus.approval import (
    CallbackApprover,
    FixedApprover,
    Gate,
    TTYApprover,
    command_prefix,
    is_known_safe,
    looks_denied,
)
from argus.config import ConfigError
from argus.executors import LocalExecutor
from argus.mock import call, final
from argus.sandbox import (
    OFF,
    READ_ONLY,
    WORKSPACE_WRITE,
    Bwrap,
    Landlock,
    NoSandbox,
    Spec,
    bwrap_works,
    choose,
    landlock_works,
)

BACKENDS = [
    pytest.param(Bwrap, marks=pytest.mark.skipif(not bwrap_works(), reason="bwrap unavailable")),
    pytest.param(
        Landlock, marks=pytest.mark.skipif(not landlock_works(), reason="landlock unavailable")
    ),
]


@pytest.fixture
def outside():
    """A directory outside the workspace and /tmp (which the sandbox leaves writable)."""
    import shutil
    import tempfile

    d = tempfile.mkdtemp(prefix="argus-outside-", dir="/var/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                pass

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield srv.getsockname()[1]
    stop.set()
    srv.close()


def run(backend, spec: Spec, cmd: str) -> subprocess.CompletedProcess:
    w = backend.wrap(["bash", "-c", cmd], spec)
    try:
        return subprocess.run(
            w.argv,
            preexec_fn=w.preexec_fn,
            pass_fds=w.pass_fds,
            capture_output=True,
            text=True,
            cwd=spec.workdir,
            timeout=30,
        )
    finally:
        if w.cleanup:
            w.cleanup()


# -- backends ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_workspace_write(backend, workspace, outside):
    b = backend()
    spec = Spec(WORKSPACE_WRITE, str(workspace), network=True)
    r = run(b, spec, f"echo x > new.txt && echo y > /tmp/argus-sbx-ok; echo z > {outside}/f")
    assert (workspace / "new.txt").read_text() == "x\n"
    assert "Read-only file system" in r.stderr or "Permission denied" in r.stderr
    assert looks_denied(r.stderr)


@pytest.mark.parametrize("backend", BACKENDS)
def test_read_only(backend, workspace):
    b = backend()
    r = run(b, Spec(READ_ONLY, str(workspace), network=True), "cat calc.py && echo x > new.txt")
    assert "def add" in r.stdout
    assert r.returncode != 0 and not (workspace / "new.txt").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_network(backend, workspace, listener):
    b = backend()
    probe = f"python3 -c 'import socket; socket.create_connection((\"127.0.0.1\", {listener}), timeout=2)'"
    on = run(b, Spec(WORKSPACE_WRITE, str(workspace), network=True), probe)
    assert on.returncode == 0, on.stderr
    off = run(b, Spec(WORKSPACE_WRITE, str(workspace), network=False), probe)
    assert off.returncode != 0


def test_choose():
    assert choose("none").name == "none"
    assert choose("auto").name in ("bwrap", "landlock", "none")


def test_executor_runs_sandboxed(workspace, outside):
    ex = LocalExecutor(str(workspace))
    ex.sandbox_backend = choose("auto")
    if ex.sandbox_backend.name == "none":
        pytest.skip("no sandbox here")
    r = ex.run(f"touch {outside}/x", sandbox=Spec(WORKSPACE_WRITE, str(workspace)))
    assert r.exit_code != 0
    assert ex.run(f"touch {outside}/x").exit_code == 0  # unsandboxed without a spec


# -- classification ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,safe",
    [
        ("ls -la", True),
        ("git status && git diff HEAD~1", True),
        ("grep -rn TODO . | head -20", True),
        ("find . -name '*.py' | wc -l", True),
        ("cat a.txt 2>&1", True),
        ("sed -n 1,20p calc.py", True),
        ("rm -rf build", False),
        ("echo hi > f.txt", False),
        ("find . -delete", False),
        ("sed -i s/a/b/ f", False),
        ("cat $(which python)", False),
        ("git commit -am x", False),
        ("python3 -m pytest -q", False),
        ("sleep 100 &", False),
    ],
)
def test_known_safe(cmd, safe):
    assert is_known_safe(cmd) is safe


def test_command_prefix():
    assert command_prefix("python3 -m pytest -q tests/") == "python3 -m pytest"
    assert command_prefix("npm test -- --watch=false") == "npm test"
    assert command_prefix("CI=1 make check") == "make"


# -- the gate -------------------------------------------------------------------------------------


class Box:
    name = "bwrap"


def gate(policy, answer="no", sandbox=True, required=False):
    asked = []

    def answer_fn(req):
        asked.append(req)
        return answer

    g = Gate(
        policy,
        Box() if sandbox else NoSandbox(),
        CallbackApprover(answer_fn),
        required=required,
    )
    return g, asked


def test_gate_read_only():
    g, asked = gate("read-only")
    assert not g.decide("edit", {"path": "a"}, True, True).allow
    d = g.decide("bash", {"cmd": "rm -rf ."}, True, True)
    assert d.allow and d.sandbox == READ_ONLY
    assert g.decide("read", {"path": "/etc/hosts"}, False, False).allow
    assert not asked and g.escalate("x") is None
    g, _ = gate("read-only", sandbox=False)
    assert not g.decide("bash", {"cmd": "ls"}, True, True).allow


def test_gate_auto():
    g, asked = gate("auto", answer="yes")
    assert g.decide("edit", {"path": "a"}, True, True).allow and not asked
    d = g.decide("edit", {"path": "/etc/x"}, True, False)
    assert d.allow and d.asked and "outside" in asked[0].reason
    d = g.decide("bash", {"cmd": "pip install x"}, True, True)
    assert d.allow and d.sandbox == WORKSPACE_WRITE
    esc = g.escalate("pip install x")
    assert esc.allow and esc.sandbox == OFF
    g, _ = gate("auto", sandbox=False)
    d = g.decide("bash", {"cmd": "make"}, True, True)
    assert d.allow and d.sandbox == OFF and d.source == "no-sandbox"
    g, _ = gate("auto", sandbox=False, required=True)
    assert not g.decide("bash", {"cmd": "make"}, True, True).allow


def test_gate_ask_remembers_always():
    g, asked = gate("ask", answer="always")
    d = g.decide("bash", {"cmd": "python3 -m pytest -q"}, True, True)
    assert d.allow and d.sandbox == WORKSPACE_WRITE and len(asked) == 1
    d = g.decide("bash", {"cmd": "python3 -m pytest -x tests/test_a.py"}, True, True)
    assert d.allow and d.source == "session" and len(asked) == 1
    d = g.decide("bash", {"cmd": "git status"}, True, True)
    assert d.allow and d.sandbox == READ_ONLY and d.source == "safe-command" and len(asked) == 1
    g, asked = gate("ask", answer="no")
    assert not g.decide("edit", {"path": "a"}, True, True).allow and len(asked) == 1


def test_gate_full():
    g, asked = gate("full")
    d = g.decide("bash", {"cmd": "rm -rf /tmp/x"}, True, True)
    assert d.allow and d.sandbox == OFF and not asked


def test_tty_approver():
    import io

    out = io.StringIO()
    answers = iter(["y", "a", "n", ""])
    a = TTYApprover(out, lambda: next(answers))
    from argus.approval import Request

    req = Request("bash", "bash: make", "why")
    assert [a.ask(req) for _ in range(4)] == ["yes", "always", "no", "no"]
    assert "bash: make" in out.getvalue()


# -- the agent ------------------------------------------------------------------------------------


needs_sandbox = pytest.mark.skipif(choose("auto").name == "none", reason="no sandbox here")


@needs_sandbox
def test_agent_read_only_policy(make_agent, workspace):
    steps = [
        call("edit", path="calc.py", old="return a - b", new="return a + b"),
        call("bash", cmd="echo hacked > calc.py; cat calc.py | head -1"),
        final("could not"),
    ]
    agent, _ = make_agent(steps, overrides=['agent.approval="read-only"'])
    result = agent.run("fix")
    assert result.status == "completed"
    assert "return a - b" in (workspace / "calc.py").read_text()
    calls = agent.store.tool_calls(result.run_id)
    assert "not allowed" in calls[0]["result"]
    assert "def add" in calls[1]["result"]  # the read part worked, the write did not
    rows = agent.store.approvals(result.run_id)
    assert [(r["tool"], r["allowed"], r["sandbox"]) for r in rows] == [
        ("edit", 0, "off"),
        ("bash", 1, "read-only"),
    ]


@needs_sandbox
def test_agent_auto_policy_escalation(make_agent, workspace, outside):
    cmd = f"echo x > {outside}/marker"
    steps = [call("bash", cmd=cmd), final("done")]
    agent, _ = make_agent(steps)  # auto policy, headless: deny
    result = agent.run("write outside")
    tc = agent.store.tool_calls(result.run_id)[0]
    assert "running it unsandboxed was not approved" in tc["result"]
    assert json_meta(tc)["sandbox"] == "workspace-write"
    rows = agent.store.approvals(result.run_id)
    assert rows[-1]["source"] == "approver" and rows[-1]["allowed"] == 0

    # with someone saying yes, it re-runs without the sandbox
    agent, _ = make_agent(steps, overrides=['agent.headless_approval="allow"'])
    result = agent.run("write outside")
    tc = agent.store.tool_calls(result.run_id)[0]
    assert "re-ran without the sandbox" in tc["result"]
    import os

    assert os.path.exists(f"{outside}/marker")


def json_meta(row):
    import json

    return json.loads(row["meta_json"])


def test_agent_ask_policy_with_callback(make_agent, workspace, mock):
    from argus.agent import Agent

    steps = [
        call("bash", cmd="ls"),
        call("edit", path="calc.py", old="return a - b", new="return a + b"),
        call("bash", cmd="python3 -c 'print(1)'"),
        final("done"),
    ]
    asked = []
    approver = CallbackApprover(lambda req: asked.append(req.summary) or "yes")
    base, server = make_agent(
        steps, overrides=['agent.approval="ask"', "tools.require_read_before_edit=false"]
    )
    agent = Agent(base.cfg, store=base.store, approver=approver)
    try:
        result = agent.run("fix")
    finally:
        agent.close()
    assert result.status == "completed"
    assert asked == ["edit: calc.py", "bash: python3 -c 'print(1)'"]  # ls is known-safe
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_required_sandbox_without_one(make_agent):
    with pytest.raises(ConfigError):
        make_agent([], overrides=['sandbox.backend="nonsense"'])
    agent, _ = make_agent(
        [call("bash", cmd="ls"), final("x")],
        overrides=['sandbox.backend="none"', "sandbox.required=true"],
    )
    result = agent.run("ls")
    assert "no sandbox is available" in agent.store.tool_calls(result.run_id)[0]["result"]
    setup = agent.store.events(result.run_id, "setup")[0]["data_json"]
    assert "commands need approval" in setup


def test_no_sandbox_auto_runs_unsandboxed(make_agent):
    agent, _ = make_agent(
        [call("bash", cmd="echo hi"), final("x")], overrides=['sandbox.backend="none"']
    )
    result = agent.run("x")
    assert "hi" in agent.store.tool_calls(result.run_id)[0]["result"]
    assert "commands run unsandboxed" in agent.store.events(result.run_id, "setup")[0]["data_json"]


def test_fixed_approver():
    assert FixedApprover("yes").ask(None) == "yes"
