"""End-to-end tool loop against the mock server (milestone 1)."""

from __future__ import annotations

import json

import pytest

from argus.mock import call, calls, final, raw_call


def fix_script():
    return [
        {"reasoning": "Look at calc.py first.", **call("read", path="calc.py")},
        {
            "reasoning": "add uses minus.",
            **call("edit", path="calc.py", old="return a - b", new="return a + b"),
        },
        call("bash", cmd="python3 -c 'import calc; assert calc.add(2, 3) == 5; print(\"ok\")'"),
        {
            "expect": {"last_contains": "ok"},
            **final("Fixed add() in calc.py; it now returns a + b."),
        },
    ]


@pytest.mark.parametrize("stream", [True, False])
def test_fix_task_end_to_end(make_agent, workspace, stream):
    agent, server = make_agent(fix_script(), [f"model.stream={str(stream).lower()}"])
    result = agent.run("Fix the add function in calc.py.")

    assert result.status == "completed", (result, server.errors)
    assert result.final.startswith("Fixed add()")
    assert "return a + b" in (workspace / "calc.py").read_text()
    assert result.turns == 4 and result.tool_calls == 3
    assert not server.errors

    st = agent.store
    run = st.run(result.run_id)
    assert run["status"] == "completed"
    assert run["protocol"] == "native"
    assert run["n_turns"] == 4
    assert run["prompt_tokens"] > 0 and run["completion_tokens"] > 0
    assert json.loads(run["config_json"])["agent"]["protocol"] == "native"

    turns = st.turns(result.run_id)
    assert [t["idx"] for t in turns] == [0, 1, 2, 3]
    assert turns[0]["reasoning"] == "Look at calc.py first."
    assert all(t["prompt_tokens"] > 0 and t["total_ms"] is not None for t in turns)
    assert turns[0]["finish_reason"] == "tool_calls" and turns[3]["finish_reason"] == "stop"
    # later turns reuse the cached prompt prefix
    assert turns[3]["cached_tokens"] > 0
    # each turn's context is the previous context plus the new messages
    ctx = [json.loads(t["context_ids"]) for t in turns]
    assert len(ctx[0]) == 2
    for a, b in zip(ctx, ctx[1:], strict=False):
        assert b[: len(a)] == a and len(b) == len(a) + 2

    tcs = st.tool_calls(result.run_id)
    assert [t["name"] for t in tcs] == ["read", "edit", "bash"]
    assert all(t["ok"] for t in tcs)
    assert "1\tdef add(a, b):" in tcs[0]["result"]
    assert json.loads(tcs[2]["meta_json"])["exit_code"] == 0

    msgs = st.messages(result.run_id)
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert st.failures(result.run_id) == []

    # the request the model saw on the last turn carries the tool results and the tools
    last = server.requests[-1]
    assert [t["function"]["name"] for t in last["tools"]] == [
        "read",
        "edit",
        "bash",
        "glob",
        "grep",
    ]
    assert last["messages"][2]["tool_calls"][0]["function"]["name"] == "read"
    assert last["messages"][3]["role"] == "tool"


def test_system_prompt_is_minimal(make_agent, workspace):
    agent, server = make_agent([final("ok")])
    agent.run("noop")
    system = server.requests[0]["messages"][0]["content"]
    assert str(workspace) in system
    assert len(system) < 400


def test_parallel_calls_in_one_turn(make_agent):
    agent, server = make_agent(
        [calls(("read", {"path": "calc.py"}), ("read", {"path": "README.md"})), final()]
    )
    r = agent.run("read two files")
    assert r.status == "completed" and r.tool_calls == 2
    req = server.requests[1]
    ids = [tc["id"] for tc in req["messages"][2]["tool_calls"]]
    assert [m["tool_call_id"] for m in req["messages"][3:5]] == ids


def test_bad_arguments_become_error_results(make_agent):
    agent, server = make_agent(
        [
            raw_call("read", '{"path": "calc.py"'),  # truncated JSON
            call("read", file="calc.py"),  # wrong parameter name
            call("teleport", to="mars"),  # unknown tool
            {"expect": {"last_contains": "unknown tool 'teleport'"}, **final("gave up")},
        ]
    )
    agent.cfg.agent.max_malformed = 5
    r = agent.run("x")
    assert r.status == "completed", server.errors
    fails = agent.store.failures(r.run_id)
    assert [f["tag"] for f in fails] == ["malformed_call"] * 3
    assert "not valid JSON" in fails[0]["detail"]
    assert "missing required path" in fails[1]["detail"]
    # history keeps a valid (empty) arguments object instead of the broken JSON
    hist = server.requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert hist == "{}"


def test_tool_error_is_returned_not_raised(make_agent):
    agent, server = make_agent(
        [
            call("read", path="missing.py"),
            {"expect": {"last_contains": "file not found"}, **final()},
        ]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    tc = agent.store.tool_calls(r.run_id)[0]
    assert tc["ok"] == 0 and "file not found" in tc["error"]


def test_max_turns(make_agent):
    agent, _ = make_agent(
        [call("glob", pattern="*.py")] * 3,
        ["agent.max_turns=3", "agent.loop_repeat=99", "agent.loop_abort=99"],
    )
    r = agent.run("x")
    assert r.status == "failed"
    assert ("max_turns", "no final answer after 3 turns") in r.failures


def test_server_error_ends_run(make_agent):
    agent, _ = make_agent([{"error": {"status": 500, "message": "kaput"}}])
    r = agent.run("x")
    assert r.status == "error"
    assert r.failures[0][0] == "server_error"
    assert agent.store.run(r.run_id)["status"] == "error"


def test_context_overflow_is_token_cap(make_agent):
    agent, _ = make_agent([final()], server_kw={"n_ctx": 20})
    r = agent.run("x")
    assert r.status == "failed"
    assert r.failures[0][0] == "token_cap" and "context overflow" in r.failures[0][1]


def test_salvaged_markup_call(make_agent, workspace):
    agent, server = make_agent(
        [
            {
                "content": 'I will read it.\n<tool_call>\n{"name": "read", "arguments": {"path": "calc.py"}}\n</tool_call>'
            },
            {"expect": {"last_role": "tool", "last_contains": "def add"}, **final()},
        ]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    tc = agent.store.tool_calls(r.run_id)
    assert tc[0]["name"] == "read" and tc[0]["ok"] == 1
