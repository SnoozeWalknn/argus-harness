from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from argus.agent import Agent
from argus.config import load_config
from argus.mock import MockServer, Script
from argus.store import Store

CALC = """def add(a, b):
    return a - b


def mul(a, b):
    return a * b
"""

TEST_CALC = """from calc import add, mul


def test_add():
    assert add(2, 3) == 5


def test_mul():
    assert mul(2, 3) == 6
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text(CALC)
    (ws / "test_calc.py").write_text(TEST_CALC)
    (ws / "README.md").write_text("# calc\n\nA tiny calculator.\n")
    (ws / "pkg").mkdir()
    (ws / "pkg" / "__init__.py").write_text("")
    (ws / "pkg" / "util.py").write_text("def helper():\n    return 'help'\n")
    return ws


def git_init(path: Path) -> None:
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")


@pytest.fixture
def mock():
    servers: list[MockServer] = []

    def make(steps=None, **kw) -> MockServer:
        script = steps if isinstance(steps, Script) else Script(steps or [])
        server = MockServer(script, **kw).start()
        servers.append(server)
        return server

    yield make
    for s in servers:
        s.stop()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "argus.db"


def q(value: str) -> str:
    return json.dumps(value)


@pytest.fixture
def make_agent(mock, workspace, db_path):
    agents: list[Agent] = []

    def make(steps=None, overrides=(), server_kw=None, workdir=None, reporter=None):
        server = mock(steps, **(server_kw or {}))
        cfg = load_config(
            None,
            [
                f"model.base_url={q(server.url)}",
                f"executor.workdir={q(str(workdir or workspace))}",
                f"log.db={q(str(db_path))}",
                "model.timeout=20",
                "model.retries=0",
                *overrides,
            ],
        )
        agent = Agent(cfg, reporter=reporter)
        agents.append(agent)
        return agent, server

    yield make
    for a in agents:
        a.close()
        a.store.close()


@pytest.fixture
def store(db_path):
    s = Store(db_path)
    yield s
    s.close()
