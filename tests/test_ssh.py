"""SSH executor: tools behave identically locally and over SSH (milestone 4).

Most tests go through ``fake_ssh.py``, which runs the remote command string
with ``sh -c`` like sshd would. ``test_real_sshd`` starts a throwaway sshd on
localhost when one is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from argus.config import ToolsConfig
from argus.executors import LocalExecutor
from argus.mock import call, final
from argus.ssh import ExecutorError, SSHExecutor
from argus.tools import ToolContext, build_tools
from tests.test_agent import fix_script

SHIM = [sys.executable, str(Path(__file__).parent / "fake_ssh.py")]


def ssh_executor(workdir, host="vm", **kw) -> SSHExecutor:
    return SSHExecutor(host, str(workdir), ssh_command=SHIM, **kw)


@pytest.fixture(params=["local", "ssh"])
def ex(request, workspace):
    e = LocalExecutor(str(workspace)) if request.param == "local" else ssh_executor(workspace)
    yield e
    e.close()


def test_run_basics(ex, workspace):
    r = ex.run("echo $((1+2)); pwd; echo err >&2; exit 4")
    assert r.exit_code == 4
    assert r.output.splitlines() == ["3", str(workspace), "err"]
    r = ex.run('printf \'%s|\' "a  b" "it\'s" \'$HOME\' "\\\\n"', timeout=10)
    assert r.output == "a  b|it's|$HOME|\\n|"  # quoting survives the round trip
    assert ex.run("cat", stdin=b"piped").output == "piped"
    assert ex.run("cat").output == ""  # no stdin: EOF, not a hang
    assert ex.run("echo $GIT_PAGER").output.strip() == "cat"


def test_run_separate_stderr(ex):
    r = ex.run("echo out; echo err >&2", merge_stderr=False)
    assert r.output == "out\n" and r.stderr == "err\n"


def test_timeout(ex):
    t0 = time.time()
    r = ex.run("sleep 20", timeout=1)
    assert r.timed_out and r.exit_code is None
    assert time.time() - t0 < 10


def test_files(ex, workspace):
    assert ex.read_bytes("calc.py").startswith(b"def add")
    assert ex.read_bytes("calc.py", max_bytes=3) == b"def"
    with pytest.raises(FileNotFoundError):
        ex.read_bytes("nope.py")
    with pytest.raises(IsADirectoryError):
        ex.read_bytes("pkg")
    ex.write_bytes("deep/new dir/f.txt", b"hello\x00world")
    assert (workspace / "deep/new dir/f.txt").read_bytes() == b"hello\x00world"
    assert ex.exists("deep/new dir/f.txt") and not ex.exists("deep/nope")
    assert "pkg/util.py" in ex.list_files()
    assert ex.has_command("sh") and not ex.has_command("definitely-not-a-command-xyz")
    assert ex.home()


def test_tools_parity(workspace, tmp_path):
    """The same tool calls give the same results through both executors."""
    other = tmp_path / "ws2"
    shutil.copytree(workspace, other)
    tools = build_tools(ToolsConfig())
    ctx_local = ToolContext(LocalExecutor(str(workspace)), ToolsConfig())
    ctx_ssh = ToolContext(ssh_executor(other), ToolsConfig())
    script = [
        ("read", {"path": "calc.py", "offset": 2}),
        ("edit", {"path": "calc.py", "old": "return a - b", "new": "return a + b"}),
        ("edit", {"path": "calc.py", "old": "  return a*b", "new": "    return b * a"}),
        ("bash", {"cmd": "python3 -c 'import calc; print(calc.add(2, 3), calc.mul(2, 3))'"}),
        ("glob", {"pattern": "**/*.py"}),
        ("grep", {"pattern": "return", "glob": "*.py"}),
        ("read", {"path": "missing.py"}),
    ]
    for name, args in script:
        outs = []
        for ctx in (ctx_local, ctx_ssh):
            try:
                outs.append(tools[name].run(ctx, dict(args)).text)
            except Exception as e:
                outs.append(f"{type(e).__name__}: {e}")
        assert outs[0] == outs[1], (name, outs)
    assert (workspace / "calc.py").read_text() == (other / "calc.py").read_text()
    assert "return b * a" in (other / "calc.py").read_text()


def test_unreachable_host_raises(workspace):
    e = ssh_executor(workspace, host="unreachable")
    with pytest.raises(ExecutorError, match="Connection refused"):
        e.run("true")


def test_agent_over_ssh(make_agent, workspace, tmp_path):
    log = tmp_path / "ssh.log"
    os.environ["FAKE_SSH_LOG"] = str(log)
    try:
        agent, server = make_agent(
            fix_script(),
            [
                'executor.kind="ssh"',
                'executor.host="vm"',
                f"executor.ssh_command={json.dumps(SHIM)}",
            ],
        )
        r = agent.run("Fix add")
    finally:
        del os.environ["FAKE_SSH_LOG"]
    assert r.status == "completed", (r, server.errors)
    assert "return a + b" in (workspace / "calc.py").read_text()
    assert agent.store.run(r.run_id)["executor"] == f"ssh:vm:{workspace}"
    assert "vm\t" in log.read_text()


def test_ssh_tool_error_does_not_crash_run(make_agent, workspace):
    agent, server = make_agent(
        [
            call("bash", cmd="true"),
            {"expect": {"last_contains": "ssh to unreachable failed"}, **final()},
        ],
        [
            'executor.kind="ssh"',
            'executor.host="unreachable"',
            f"executor.ssh_command={json.dumps(SHIM)}",
        ],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors


# -- a real sshd -------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sshd(tmp_path_factory):
    sshd_bin = shutil.which("sshd") or "/usr/sbin/sshd"
    if not os.path.exists(sshd_bin) or not shutil.which("ssh") or not shutil.which("ssh-keygen"):
        pytest.skip("openssh not installed")
    d = tmp_path_factory.mktemp("sshd")
    for name in ("host_key", "client_key"):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / name)], check=True
        )
    (d / "authorized_keys").write_text((d / "client_key.pub").read_text())
    port = _free_port()
    (d / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {d / 'host_key'}\n"
        f"AuthorizedKeysFile {d / 'authorized_keys'}\nPidFile {d / 'sshd.pid'}\n"
        "StrictModes no\nUsePAM no\nPasswordAuthentication no\nPermitRootLogin prohibit-password\n"
    )
    os.makedirs("/run/sshd", exist_ok=True)
    proc = subprocess.Popen(
        [sshd_bin, "-D", "-e", "-f", str(d / "sshd_config")], stderr=subprocess.PIPE
    )
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            if proc.poll() is not None:
                pytest.skip(f"sshd did not start: {proc.stderr.read().decode()[:200]}")
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("sshd did not start")
    user = os.environ.get("USER") or "root"
    yield {
        "host": f"{user}@127.0.0.1",
        "options": [
            "-p", str(port), "-i", str(d / "client_key"),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        ],
    }  # fmt: skip
    proc.terminate()
    proc.wait(5)


def test_real_sshd(sshd, workspace, make_agent):
    e = SSHExecutor(sshd["host"], str(workspace), ssh_options=sshd["options"])
    try:
        r = e.run("echo hi; pwd")
        if r.exit_code != 0:
            pytest.skip(f"ssh login failed: {r.output[:200]}")
        assert r.output.splitlines() == ["hi", str(workspace)]
        t0 = time.perf_counter()
        for _ in range(5):
            e.run("true")
        per_call = (time.perf_counter() - t0) / 5
        assert per_call < 1.0  # control socket reuse keeps calls cheap
        assert e.run("sleep 30", timeout=1).timed_out
    finally:
        e.close()
    agent, server = make_agent(
        fix_script(),
        [
            'executor.kind="ssh"',
            f"executor.host={json.dumps(sshd['host'])}",
            f"executor.ssh_options={json.dumps(sshd['options'])}",
        ],
    )
    r = agent.run("Fix add")
    assert r.status == "completed", (r, server.errors)
    assert "return a + b" in (workspace / "calc.py").read_text()
