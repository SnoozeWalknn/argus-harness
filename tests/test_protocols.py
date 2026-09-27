"""The same scenarios under every tool-call protocol (milestone 2)."""

from __future__ import annotations

import json

import pytest

from argus.config import AgentConfig, ToolsConfig
from argus.llm import Completion, RawToolCall
from argus.mock import call, final
from argus.protocols import JsonSchemaProtocol, NativeProtocol, make_protocol, salvage_calls
from argus.tools import build_tools
from tests.test_agent import fix_script

PROTOCOLS = ["native", "json_schema", "grammar"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [True, False])
def test_fix_task_under_each_protocol(make_agent, workspace, protocol, stream):
    agent, server = make_agent(
        fix_script(),
        [f"agent.protocol={json.dumps(protocol)}", f"model.stream={str(stream).lower()}"],
    )
    r = agent.run("Fix add in calc.py")
    assert r.status == "completed", (r, server.errors)
    assert not server.errors  # scripted output satisfied the request's schema/grammar
    assert "return a + b" in (workspace / "calc.py").read_text()
    assert r.final.startswith("Fixed add()")
    assert [t["name"] for t in agent.store.tool_calls(r.run_id)] == ["read", "edit", "bash"]

    req = server.requests[0]
    system = req["messages"][0]["content"]
    if protocol == "native":
        assert "tools" in req and "grammar" not in req and "response_format" not in req
    else:
        assert "tools" not in req
        assert "read(path, offset?, limit?) - show a file with line numbers" in system
        assert "done(summary)" in system
        # results come back wrapped the way Qwen templates render tool output
        assert server.requests[1]["messages"][-1]["content"].startswith(
            "<tool_response>\n1\tdef add"
        )
    if protocol == "json_schema":
        assert req["response_format"]["type"] == "json_schema"
        assert req["chat_template_kwargs"] == {"enable_thinking": False}
    if protocol == "grammar":
        assert req["grammar"].startswith("root ::= think? action")


def test_json_schema_thinking_override(make_agent):
    agent, server = make_agent(
        [final()], ['agent.protocol="json_schema"', "model.enable_thinking=true"]
    )
    agent.run("x")
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("protocol", ["json_schema", "grammar"])
def test_thought_field(make_agent, protocol):
    agent, server = make_agent(
        [{"thought": "look first", **call("glob", pattern="*.py")}, final("ok")],
        [f"agent.protocol={json.dumps(protocol)}", "agent.json_thought=true"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert not server.errors
    assert '"thought"' in server.requests[0]["messages"][0]["content"]
    turn0 = agent.store.turns(r.run_id)[0]
    assert json.loads(turn0["content"])["thought"] == "look first"


@pytest.mark.parametrize("protocol", ["json_schema", "grammar"])
def test_constrained_malformed_output(make_agent, protocol):
    agent, server = make_agent(
        [
            {"raw": '{"tool": "read", "args": {"path": '},  # cut-off JSON
            {"raw": "[1, 2]"},
            {"raw": ""},
            {"expect": {"last_contains": "empty reply"}, **final("ok")},
        ],
        [f"agent.protocol={json.dumps(protocol)}", "agent.max_malformed=10"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    tags = [t for t, _ in r.failures]
    assert tags == ["malformed_call"] * 3
    assert "not a valid JSON action" in r.failures[0][1]
    msgs = server.requests[1]["messages"]
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith("Error:")


def test_constrained_parse_strips_think_block():
    proto = JsonSchemaProtocol(build_tools(ToolsConfig()), AgentConfig())
    c = Completion(content='<think>hmm</think>\n{"tool":"read","args":{"path":"a"}}')
    p = proto.parse(c, 0)
    assert p.calls[0].name == "read" and p.calls[0].args == {"path": "a"}
    assert p.assistant["content"] == '{"tool":"read","args":{"path":"a"}}'
    assert p.content == ""  # no thought: nothing to show besides the call itself
    p = proto.parse(Completion(content='{"tool":"done","args":{"summary":"fin"}}'), 1)
    assert p.final == "fin" and not p.calls
    p = proto.parse(Completion(content='{"tool":"read","args":{"path":1}}'), 2)
    assert "path must be string" in p.calls[0].error


def test_native_parse_problem_without_salvage():
    proto = NativeProtocol(build_tools(ToolsConfig()), AgentConfig())
    p = proto.parse(Completion(content="<tool_call>\n{broken"), 0)
    assert p.problems and not p.calls and p.final is None


def test_salvage_formats():
    hermes = '<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>'
    assert [(c.name, c.args) for c in salvage_calls(hermes, 0)] == [("read", {"path": "a.py"})]
    xml = "<tool_call>\n<function=bash>\n<parameter=cmd>\nls -la\n</parameter>\n<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call>"
    assert [(c.name, c.args) for c in salvage_calls(xml, 0)] == [
        ("bash", {"cmd": "ls -la", "timeout": 5})
    ]


def test_native_coerces_stringly_args():
    proto = NativeProtocol(build_tools(ToolsConfig()), AgentConfig())
    c = Completion(
        tool_calls=[RawToolCall("c1", "read", '{"path": "a", "offset": "3", "limit": null}')]
    )
    p = proto.parse(c, 0)
    assert p.calls[0].error is None
    assert p.calls[0].args == {"path": "a", "offset": 3}


def test_make_protocol_unknown():
    with pytest.raises(ValueError, match="unknown protocol"):
        make_protocol("smoke-signals", {}, AgentConfig())
