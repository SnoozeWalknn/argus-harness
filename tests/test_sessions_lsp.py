"""Sessions (continue, resume, switch model/protocol/provider) and LSP diagnostics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from argus.cli import main
from argus.config import LSPConfig
from argus.executors import LocalExecutor
from argus.lsp import Diagnostics
from argus.mock import MockServer, Script, call, final
from argus.session import to_constrained, to_native
from argus.ssh import SSHExecutor

from .conftest import q

FAKE_LSP = [sys.executable, str(Path(__file__).parent / "fake_lsp.py")]
SHIM = [sys.executable, str(Path(__file__).parent / "fake_ssh.py")]


def roles(req) -> list[str]:
    return [m["role"] for m in req["messages"]]


# -- sessions ------------------------------------------------------------------------------------


def test_second_run_continues_the_conversation(make_agent):
    steps = [call("read", path="calc.py"), final("add subtracts."), final("mul is fine.")]
    agent, server = make_agent(steps)
    first = agent.run("what is wrong with add?")
    sid = agent.store.run(first.run_id)["session_id"]
    second = agent.run("and mul?", session=sid)
    req = server.requests[2]
    assert roles(req) == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert req["messages"][1]["content"] == "what is wrong with add?"
    assert req["messages"][4]["content"] == "add subtracts."
    assert req["messages"][-1]["content"] == "and mul?"
    sess = agent.store.session(sid)
    assert (sess["n_runs"], sess["last_run_id"]) == (2, second.run_id)
    assert agent.store.run(second.run_id)["session_id"] == sid
    # the continued run references the earlier messages instead of copying them
    turn = agent.store.turns(second.run_id)[0]
    ids = json.loads(turn["context_ids"])
    assert ids[1] in [m["id"] for m in agent.store.messages(first.run_id)]


def test_cli_continue_and_sessions(tmp_path, workspace, capsys):
    server = MockServer(Script([final("first answer"), final("second answer")])).start()
    common = ["-w", str(workspace), "-o", f"model.base_url={q(server.url)}", "--db", str(tmp_path / "a.db"), "-q"]  # fmt: skip
    try:
        assert main(["run", "first question", *common]) == 0
        assert main(["run", "--continue", "second question", *common]) == 0
    finally:
        server.stop()
    assert roles(server.requests[1])[-3:] == ["user", "assistant", "user"]
    capsys.readouterr()
    assert main(["sessions", "--db", str(tmp_path / "a.db")]) == 0
    out = capsys.readouterr().out
    assert "first question" in out and " 2 " in out


def test_continue_without_a_session_is_an_error(tmp_path, workspace, capsys):
    rc = main(["run", "--continue", "x", "-w", str(workspace), "--db", str(tmp_path / "a.db")])
    assert rc == 2 and "no session" in capsys.readouterr().err


def test_batch_runs_have_no_session(make_agent):
    agent, _ = make_agent([final()])
    r = agent.run("x", batch_id=None, task_id="t")
    assert agent.store.run(r.run_id)["session_id"]  # interactive: yes
    b = agent.store.new_batch("suite")
    agent2, _ = make_agent([final()])
    r2 = agent2.run("x", batch_id=b)
    assert agent2.store.run(r2.run_id)["session_id"] is None


# -- protocol conversion ------------------------------------------------------------------------------


NATIVE_HISTORY = [
    {"role": "user", "content": "fix"},
    {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}},
            {"id": "c2", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}},
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "A"},
    {"role": "tool", "tool_call_id": "c2", "content": "B"},
    {"role": "assistant", "content": "Done."},
]  # fmt: skip


def test_convert_native_to_constrained_and_back():
    constrained = to_constrained(NATIVE_HISTORY)
    assert [m["role"] for m in constrained] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]  # fmt: skip
    assert json.loads(constrained[1]["content"]) == {"tool": "read", "args": {"path": "a"}}
    assert constrained[2]["content"] == "<tool_response>\nA\n</tool_response>"
    native = to_native(constrained)
    assert [m["role"] for m in native] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert native[1]["tool_calls"][0]["function"]["name"] == "read"
    assert native[2] == {
        "role": "tool",
        "tool_call_id": native[1]["tool_calls"][0]["id"],
        "content": "A",
    }
    done = to_native([{"role": "assistant", "content": '{"tool":"done","args":{"summary":"ok"}}'}])
    assert done == [{"role": "assistant", "content": "ok"}]


def test_switch_protocol_mid_session(make_agent):
    steps = [call("read", path="calc.py"), final("add subtracts"), final("ok")]
    agent, server = make_agent(steps, overrides=['agent.protocol="grammar"'])
    first = agent.run("look at calc.py")
    sid = agent.store.run(first.run_id)["session_id"]
    native, _ = make_agent([], overrides=['agent.protocol="native"'])
    native.llm = agent.llm  # same server
    native.store = agent.store
    second = native.run("thanks", session=sid)
    req = server.requests[2]
    assert roles(req) == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert req["messages"][2]["tool_calls"][0]["function"]["name"] == "read"
    ev = json.loads(native.store.events(second.run_id, "session")[0]["data_json"])
    assert ev["converted"] == "grammar -> native"
    kinds = {m["kind"] for m in native.store.messages(second.run_id)}
    assert "history" in kinds


# -- switching providers ---------------------------------------------------------------------------


def test_switch_from_llama_to_anthropic_mid_session(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    steps = [
        call("read", path="calc.py"),
        final("add subtracts"),
        {"reasoning": "ok", **final("done")},
    ]
    local, server = make_agent(steps)
    first = local.run("look at calc.py")
    sid = local.store.run(first.run_id)["session_id"]
    claude, _ = make_agent(
        [], overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"']
    )
    claude.llm.base_url = server.url  # the same mock serves /v1/messages
    claude.store = local.store
    second = claude.run("now fix it", session=sid)
    assert second.status == "completed", second.failures
    assert server.errors == []  # tool_use / tool_result pairing held up
    wire = server.requests[2]
    blocks = [b["type"] for m in wire["messages"] for b in m["content"]]
    assert blocks == ["text", "tool_use", "tool_result", "text", "text"]
    ev = json.loads(claude.store.events(second.run_id, "session")[0]["data_json"])
    assert ev["switched"]["to"][1] == "claude-sonnet-5"


def test_same_anthropic_model_keeps_thinking_across_runs(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    steps = [{"reasoning": "first thoughts", **final("one")}, final("two")]
    agent, server = make_agent(
        steps, overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"']
    )
    first = agent.run("q1")
    sid = agent.store.run(first.run_id)["session_id"]
    second = agent.run("q2", session=sid)
    assert second.status == "completed" and server.errors == []
    replayed = server.requests[1]["messages"][1]["content"]
    assert replayed[0]["type"] == "thinking" and replayed[0]["signature"].startswith("mocksig.")


def test_switch_from_anthropic_to_gemini_3(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza0000000000")
    steps = [
        {"reasoning": "read it", **call("read", path="calc.py")},
        final("add subtracts"),
        final("fixed plan"),
    ]
    claude, server = make_agent(
        steps, overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"']
    )
    first = claude.run("look")
    sid = claude.store.run(first.run_id)["session_id"]
    gem, _ = make_agent(
        [], overrides=['model.provider="gemini"', 'model.model="gemini-3-pro-preview"']
    )
    gem.llm.base_url = server.url
    gem.store = claude.store
    second = gem.run("plan the fix", session=sid)
    assert second.status == "completed", second.failures
    assert server.errors == []
    contents = server.requests[2]["contents"]
    call_part = contents[1]["parts"][0]
    assert call_part["functionCall"]["name"] == "read"
    assert call_part["thoughtSignature"] == "context_engineering_is_the_way_to_go"


def test_refused_then_continued_on_another_model(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    claude, _ = make_agent(
        [call("read", path="calc.py"), {"content": "", "refusal": {"category": "cyber"}}],
        overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"'],
    )
    first = claude.run("look at calc.py")
    assert first.status == "refused"
    sid = claude.store.run(first.run_id)["session_id"]
    local, server = make_agent([final("add subtracts")])
    local.store = claude.store
    second = local.run("go on", session=sid)
    assert second.status == "completed" and server.errors == []
    req = server.requests[0]
    # the tool work before the refusal carries over; the refused reply does not
    assert roles(req) == ["system", "user", "assistant", "tool", "user"]
    assert req["messages"][-1]["content"] == "go on"


def test_cli_exit_status_for_a_refusal(tmp_path, workspace, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    server = MockServer(Script([{"content": "", "refusal": {"category": "cyber"}}])).start()
    args = ["-m", "anthropic/claude-opus-5", "-o", f"model.base_url={q(server.url)}"]
    common = ["-w", str(workspace), "--db", str(tmp_path / "a.db"), *args]
    try:
        assert main(["run", "x", *common]) == 3
    finally:
        server.stop()
    out, err = capsys.readouterr()
    assert out.strip() == "The request was declined."  # the reason is the answer
    assert "the model declined" in err and "argus run --continue" in err
    assert "refused" in err


# -- LSP diagnostics --------------------------------------------------------------------------------


def lsp_config(**kw) -> LSPConfig:
    return LSPConfig(servers={"python": FAKE_LSP}, wait_ms=5000, **kw)


def test_diagnostics_with_the_fake_server(workspace):
    d = Diagnostics(LocalExecutor(str(workspace)), lsp_config())
    try:
        path = str(workspace / "calc.py")
        report = d.check(path, "def add(a, b):\n    return undefined_name  # TODO\n")
        assert [x.message for x in report.errors()] == ['"undefined_name" is not defined']
        assert len(report.errors(warnings=True)) == 2
        assert report.errors()[0].line == 2 and report.errors()[0].col == 12
        clean = d.check(path, "def add(a, b):\n    return a + b\n")
        assert clean.errors() == [] and not clean.timed_out
        assert len(d.clients) == 1  # one server for the workspace
        assert d.check(str(workspace / "README.md"), "# hi") is None  # no server for markdown
    finally:
        d.close()


def test_edit_result_carries_diagnostics(make_agent, workspace):
    steps = [
        call("read", path="calc.py"),
        call("edit", path="calc.py", old="return a - b", new="return undefined_name"),
        call("edit", path="calc.py", old="return undefined_name", new="return a + b"),
        final("fixed"),
    ]
    agent, _ = make_agent(
        steps,
        overrides=["lsp.enabled=true", f"lsp.servers={{python = {json.dumps(FAKE_LSP)}}}"],
    )
    result = agent.run("fix add")
    bad, good = agent.store.tool_calls(result.run_id)[1:3]
    assert (
        'reports after this edit:\ncalc.py:2:12 error: "undefined_name" is not defined (fake-lsp)'
        in bad["result"]
    )
    assert "reports after this edit" not in good["result"]
    events = agent.store.events(result.run_id, "diagnostics")
    assert len(events) == 2 and json.loads(events[1]["data_json"])["items"] == []


def test_diagnostics_over_ssh(workspace):
    ex = SSHExecutor("fakehost", str(workspace), ssh_command=SHIM)
    d = Diagnostics(ex, lsp_config())
    try:
        report = d.check(str(workspace / "calc.py"), "x = undefined_name\n")
        assert report is not None and len(report.errors()) == 1
    finally:
        d.close()


def test_missing_server_is_quiet(workspace):
    d = Diagnostics(
        LocalExecutor(str(workspace)), LSPConfig(servers={"python": ["no-such-lsp-binary"]})
    )
    assert d.check(str(workspace / "calc.py"), "x = 1\n") is None
    assert "no-such-lsp-binary" in d.failed["python"]


PYLSP = shutil.which("pylsp") or str(Path(sys.executable).parent / "pylsp")


@pytest.mark.skipif(not Path(PYLSP).exists(), reason="python-lsp-server not installed")
def test_real_pylsp(workspace):
    d = Diagnostics(
        LocalExecutor(str(workspace)), LSPConfig(servers={"python": [PYLSP]}, wait_ms=15000)
    )
    try:
        report = d.check(str(workspace / "calc.py"), "def add(a, b):\n    return a + c\n")
        assert report is not None and not report.timed_out
        assert any("undefined name 'c'" in x.message for x in report.errors(warnings=True))
    finally:
        d.close()


def test_files_read_earlier_in_the_session_can_be_edited(make_agent, workspace):
    steps = [
        call("read", path="calc.py"),
        final("add subtracts"),
        call("edit", path="calc.py", old="return a - b", new="return a + b"),  # no re-read
        final("fixed"),
    ]
    agent, _ = make_agent(steps)
    first = agent.run("what is wrong with add?")
    sid = agent.store.run(first.run_id)["session_id"]
    second = agent.run("fix it", session=sid)
    assert second.status == "completed", second.failures
    edit = agent.store.tool_calls(second.run_id)[0]
    assert edit["ok"], edit["result"]
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_a_file_changed_since_it_was_read_is_flagged(make_agent, workspace):
    steps = [
        call("read", path="calc.py"),
        final("seen"),
        call("edit", path="calc.py", old="return max(a, b)", new="return b * a"),
        final("done"),
    ]
    agent, _ = make_agent(steps)
    first = agent.run("look")
    (workspace / "calc.py").write_text((workspace / "calc.py").read_text() + "\n# touched\n")
    second = agent.run("swap", session=agent.store.run(first.run_id)["session_id"])
    edit = agent.store.tool_calls(second.run_id)[0]
    assert not edit["ok"] and "the file changed since you last read it" in edit["result"]
