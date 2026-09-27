"""OpenAI Responses API adapter, against the mock server's /v1/responses."""

from __future__ import annotations

import json

import pytest

from argus.mock import call, final
from argus.providers import ProviderOptions, make_provider
from argus.providers.base import ContextOverflow, LLMError
from argus.providers.openai import OpenAIProvider

KEY = "sk-proj-test-0000"
READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Show a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}
NO_SAMPLING = ProviderOptions(quirks=frozenset({"no_sampling"}))


def openai(server=None, model="gpt-5", **kw) -> OpenAIProvider:
    base = f"{server.root}/v1" if server else ""
    kw.setdefault("retries", 0)
    return make_provider("openai", base_url=base, model=model, api_key=KEY, **kw)


def test_wire_body_translation():
    body = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "You are argus."},
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "Looking.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "A"},
        ],
        "tools": [READ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 16000,
        "temperature": 0.2,
        "top_k": 20,
        "thinking": {"effort": "high"},
    }  # fmt: skip
    w = openai(options=NO_SAMPLING).wire_body(body, stream=True)
    assert w["instructions"] == "You are argus."
    assert w["input"] == [
        {"role": "user", "content": "fix it"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": '{"path":"a"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "A"},
    ]  # fmt: skip
    assert w["tools"][0] == {
        "type": "function",
        "name": "read",
        "description": "Show a file",
        "parameters": READ["function"]["parameters"],
        "strict": False,
    }
    assert w["max_output_tokens"] == 16000
    assert w["reasoning"] == {"effort": "high", "summary": "auto"}
    assert w["include"] == ["reasoning.encrypted_content"] and w["store"] is False
    assert "temperature" not in w and "top_k" not in w
    off = openai().wire_body({"messages": [], "thinking": {"enabled": False}}, False)
    assert off["reasoning"] == {"effort": "minimal"}
    plain = openai(model="gpt-4.1", options=ProviderOptions(thinking="none"))
    w = plain.wire_body({"messages": [], "temperature": 0.2}, False)
    assert "reasoning" not in w and "include" not in w and w["temperature"] == 0.2


def test_replay_items_bound_to_model():
    items = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc", "status": "completed"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{}", "status": "completed"},
    ]  # fmt: skip
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            "replay": {"provider": "openai", "model": "gpt-5", "items": items},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "A"},
    ]  # fmt: skip
    same = openai().wire_body({"model": "gpt-5", "messages": msgs}, False)["input"]
    assert same[1] == {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}
    assert same[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read",
        "arguments": "{}",
    }
    other = openai().wire_body({"model": "gpt-5-mini", "messages": msgs}, False)["input"]
    assert [i.get("type") for i in other] == [None, "function_call", "function_call_output"]
    assert OpenAIProvider.strip_replay(msgs[1]["replay"])["items"] == [items[1]]


@pytest.mark.parametrize("stream", [True, False])
def test_chat_reasoning_summary_and_function_call(mock, stream):
    s = mock([{"reasoning": "look at a.py", "content": "Reading.", **call("read", path="a.py")}])
    c = openai(s).chat(
        {"model": "gpt-5", "messages": [{"role": "user", "content": "go"}], "tools": [READ], "max_tokens": 900},
        stream=stream,
    )  # fmt: skip
    assert s.errors == []
    assert c.reasoning == "look at a.py" and c.content == "Reading."
    assert [(t.name, json.loads(t.arguments)) for t in c.tool_calls] == [("read", {"path": "a.py"})]
    assert c.tool_calls[0].id.startswith("call_mock_")
    assert c.finish_reason == "tool_calls"
    assert c.reported_reasoning_tokens == 7  # "look", " ", "at", " ", "a", ".", "py"
    assert [i["type"] for i in c.replay["items"]] == ["reasoning", "message", "function_call"]
    assert c.replay["items"][0]["encrypted_content"].startswith("mockenc.gpt-5.")


def test_agent_run_replays_encrypted_reasoning(make_agent, workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    steps = [
        {"reasoning": "read first", **call("read", path="calc.py")},
        {
            "reasoning": "fix",
            **call("edit", path="calc.py", old="return a - b", new="return a + b"),
        },
        final("Fixed."),
    ]
    agent, server = make_agent(
        steps,
        overrides=[
            'model.provider="openai"',
            'model.model="gpt-5"',
            'model.quirks=["no_sampling"]',
        ],
    )
    result = agent.run("fix add")
    assert result.status == "completed", result.failures
    assert server.errors == []
    assert "return a + b" in (workspace / "calc.py").read_text()
    second = server.requests[1]["input"]
    assert [i.get("type") for i in second][1:4] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    run = agent.store.run(result.run_id)
    assert run["reasoning_tokens"] > 0


def test_sampling_on_reasoning_model_is_rejected_without_quirk(mock):
    s = mock([final()])
    with pytest.raises(LLMError, match="temperature"):
        openai(s).chat({"model": "gpt-5", "messages": [{"role": "user", "content": "x"}], "temperature": 0.2})  # fmt: skip


def test_errors_and_limits(mock):
    s = mock(
        [
            {"error": {"status": 429, "message": "Rate limit", "headers": {"retry-after": "0"}}},
            final("ok"),
            {"content": "word " * 50},
            {"content": "", "refusal": "No."},
        ]
    )
    p = openai(s, retries=1)
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "x"}], "max_tokens": 10}
    assert p.chat(body).content == "ok"
    c = p.chat(body)
    assert c.finish_reason == "length" and c.stop_raw == "max_output_tokens"
    c = p.chat({**body, "max_tokens": 100}, stream=False)
    assert c.finish_reason == "refusal" and c.refusal == "No."
    with pytest.raises(ContextOverflow):
        openai(mock([final()], n_ctx=3)).chat(body)


def test_stream_error_event(mock):
    s = mock([{"content": "partial", "stream_error": True}])
    with pytest.raises(LLMError, match="server_error"):
        openai(s).chat({"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]})


def test_missing_key(mock):
    s = mock([final()])
    p = make_provider("openai", base_url=f"{s.root}/v1", model="gpt-5", require_key=False)
    with pytest.raises(LLMError, match="401"):
        p.chat({"messages": [{"role": "user", "content": "x"}]})
