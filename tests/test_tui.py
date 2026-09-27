"""The Textual TUI, driven with Textual's Pilot against the mock server."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("textual")

from argus.cli import model_overrides  # noqa: E402
from argus.config import load_config  # noqa: E402
from argus.mock import call, final  # noqa: E402
from argus.tui import themes  # noqa: E402
from argus.tui.app import ArgusApp  # noqa: E402

from .conftest import q  # noqa: E402


def factory(server, workspace, db_path, *extra):
    def make(spec):
        overrides = [
            f"model.base_url={q(server.url)}",
            f"executor.workdir={q(str(workspace))}",
            f"log.db={q(str(db_path))}",
            "model.retries=0",
            "lsp.enabled=false",
            *extra,
        ]
        return load_config(None, (model_overrides(spec) if spec else []) + overrides)

    return make


def texts(app, selector: str) -> list[str]:
    out = []
    for w in app.query(selector):
        content = getattr(w, "content", None)
        if content is None and hasattr(w, "source"):
            content = w.source
        out.append(str(getattr(content, "plain", content)))
    return out


async def until(pilot, cond, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("timed out waiting for the TUI")
        await pilot.pause(0.05)


async def send(app, pilot, text: str) -> None:
    done = app.runs_done
    app.query_one("#prompt").value = text
    await pilot.press("enter")
    await until(pilot, lambda: app.runs_done > done)
    await pilot.pause(0.1)
    app.drain()


def run(coro) -> None:
    asyncio.run(coro)


def test_runs_a_task_and_shows_everything(mock, workspace, db_path):
    todo = [{"content": "fix add", "status": "in_progress"}]
    server = mock(
        [
            {"reasoning": "read it first", **call("read", path="calc.py")},
            call("todo", items=todo),
            call("edit", path="calc.py", old="return a - b", new="return a + b"),
            final("Fixed **add**."),
        ]
    )
    app = ArgusApp(factory(server, workspace, db_path, "tools.todo='on'"))

    async def go():
        async with app.run_test(size=(140, 45)) as pilot:
            status = str(app.query_one("#status").content)
            assert "argus" in status and "BUILD" in status
            await send(app, pilot, "fix add")
            assert any("› fix add" in t for t in texts(app, ".user"))
            assert any("read it first" in t for t in texts(app, ".reasoning"))
            tools = texts(app, ".tool")
            assert any(t.startswith("▸ read(path=calc.py) ✓") for t in tools)
            assert any("edit(" in t and "✓" in t for t in tools)
            assert "Fixed **add**." in texts(app, ".final")[0]
            assert "[>] fix add" in str(app.query_one("#todo").content)
            assert "calc.py" in str(app.query_one("#changed").content)
            assert app.session_id and app.session_id.startswith("s")

    run(go())
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_second_message_continues_the_session(mock, workspace, db_path):
    server = mock([final("one"), final("two")])
    app = ArgusApp(factory(server, workspace, db_path))

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot, "first")
            sid = app.session_id
            await send(app, pilot, "second")
            assert app.session_id == sid

    run(go())
    msgs = server.requests[1]["messages"]
    assert [m["content"] for m in msgs[1:]] == ["first", "one", "second"]


def test_tab_toggles_plan_mode(mock, workspace, db_path):
    server = mock([final("1. change it")])
    app = ArgusApp(factory(server, workspace, db_path))

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("tab")
            assert app.mode == "plan" and "PLAN" in str(app.query_one("#status").content)
            await send(app, pilot, "how would you fix add?")
            await pilot.press("tab")
            assert app.mode == "build"

    run(go())
    tools = sorted(t["function"]["name"] for t in server.requests[0]["tools"])
    assert "edit" not in tools
    assert "plan mode" in server.requests[0]["messages"][0]["content"]


def test_approval_modal(mock, workspace, db_path):
    steps = [
        call("edit", path="calc.py", old="return a - b", new="return a + b"),
        final("done"),
    ]
    server = mock(steps)
    make = factory(
        server, workspace, db_path, 'agent.approval="ask"', "tools.require_read_before_edit=false"
    )
    app = ArgusApp(make)

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#prompt").value = "fix add"
            await pilot.press("enter")
            await until(pilot, lambda: len(app.screen.query("#approval-summary")) == 1)
            assert "edit: calc.py" in str(app.screen.query_one("#approval-summary").content)
            await pilot.press("y")
            await until(pilot, lambda: app.runs_done == 1)

    run(go())
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_switch_model_mid_session(mock, workspace, db_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-000000000")
    server = mock([final("local answer"), final("claude answer")])
    app = ArgusApp(factory(server, workspace, db_path))

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot, "question one")
            await pilot.press("ctrl+o")
            await until(pilot, lambda: len(app.screen.query("#picker-input")) == 1)
            app.screen.query_one("#picker-input").value = "anthropic/claude-sonnet-5"
            await pilot.press("enter")
            await until(pilot, lambda: len(app.screen_stack) == 1)
            assert "claude-sonnet-5" in str(app.query_one("#status").content)
            await send(app, pilot, "question two")

    run(go())
    wire = server.requests[1]
    assert server.paths[1] == "/v1/messages"
    texts_sent = [b.get("text") for m in wire["messages"] for b in m["content"]]
    assert texts_sent == ["question one", "local answer", "question two"]


def test_theme_cycles_and_is_remembered(mock, workspace, db_path, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    server = mock([])
    app = ArgusApp(factory(server, workspace, db_path))
    seen = []

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            seen.append(app.theme)
            await pilot.press("ctrl+t")
            seen.append(app.theme)
            await pilot.pause(0.05)
            assert seen[1] in str(app.query_one("#session").content)

    run(go())
    assert seen[0] == "tokyo-night" and seen[1] != seen[0]
    assert json.loads((tmp_path / "state/argus/tui.json").read_text())["theme"] == seen[1]
    assert themes.initial("auto", {seen[1], "tokyo-night"}) == seen[1]


def test_follows_omarchy_theme(tmp_path, monkeypatch):
    home = tmp_path / "h"
    (home / ".config/omarchy/themes/kanagawa").mkdir(parents=True)
    (home / ".config/omarchy/current").mkdir(parents=True)
    (home / ".config/omarchy/current/theme").symlink_to(home / ".config/omarchy/themes/kanagawa")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "nostate"))
    assert themes.omarchy_theme() == "kanagawa"
    assert themes.initial("auto", {"kanagawa", "tokyo-night"}) == "kanagawa"
    assert themes.initial("nord", {"kanagawa", "nord"}) == "nord"


def test_escape_interrupts(mock, workspace, db_path):
    server = mock([{"reasoning": "thinking ", "reasoning_repeat": 2000, "chunk_delay": 0.01}])
    app = ArgusApp(factory(server, workspace, db_path))

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#prompt").value = "think forever"
            await pilot.press("enter")
            await until(pilot, lambda: any(texts(app, ".reasoning")))
            await pilot.press("escape")
            await until(pilot, lambda: app.runs_done == 1)
            app.drain()
            assert any("run interrupted" in t for t in texts(app, ".failure"))

    run(go())


def test_sessions_picker_reopens_a_session(mock, workspace, db_path):
    server = mock([final("remembered")])
    app = ArgusApp(factory(server, workspace, db_path))

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot, "remember this")
            sid = app.session_id
            await pilot.press("ctrl+n")
            assert app.session_id is None and not texts(app, ".user")
            await pilot.press("ctrl+s")
            await until(
                pilot, lambda: app.screen.focused and app.screen.focused.id == "picker-list"
            )
            await pilot.press("enter")
            await until(pilot, lambda: len(app.screen_stack) == 1)
            assert app.session_id == sid
            assert any("remember this" in t for t in texts(app, ".user"))

    run(go())
