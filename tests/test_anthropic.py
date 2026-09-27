"""Anthropic Messages API adapter, against the mock server's /v1/messages."""

from __future__ import annotations

import json

import pytest

from argus.config import ConfigError
from argus.mock import call, final
from argus.providers import ProviderOptions, make_provider
from argus.providers.anthropic import AnthropicProvider, normalise_usage
from argus.providers.base import ContextOverflow, LLMError
from argus.tokens import TokenCounter, measure

KEY = "sk-ant-test-0000"
READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Show a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


def anthropic(server=None, **kw) -> AnthropicProvider:
    base = f"{server.root}/v1" if server else ""
    kw.setdefault("retries", 0)
    return make_provider("anthropic", base_url=base, model="claude-opus-5", api_key=KEY, **kw)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)


# -- request translation ------------------------------------------------------------------


def test_wire_body_translation():
    p = anthropic()
    body = {
        "model": "claude-opus-5",
        "messages": [
            {"role": "system", "content": "You are argus."},
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "Looking.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}},
                    {"id": "call.2", "type": "function", "function": {"name": "read", "arguments": "{bad"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "A"},
            {"role": "tool", "tool_call_id": "call.2", "content": ""},
            {"role": "user", "content": "Error: loop detected"},
        ],
        "tools": [READ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tokens": 4000,
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "thinking": {"effort": "high"},
    }  # fmt: skip
    w = p.wire_body(body, stream=True)
    assert w["system"] == [
        {"type": "text", "text": "You are argus.", "cache_control": {"type": "ephemeral"}}
    ]
    assert w["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "fix it"}]}
    assert w["messages"][1]["content"] == [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "a"}},
        {"type": "tool_use", "id": "call_2", "name": "read", "input": {}},
    ]
    assert w["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "A"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "(no output)"},
        {
            "type": "text",
            "text": "Error: loop detected",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert w["tools"] == [
        {
            "name": "read",
            "description": "Show a file",
            "input_schema": READ["function"]["parameters"],
        }
    ]
    assert w["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert w["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert w["output_config"] == {"effort": "high"}
    # thinking is on: temperature and top_k go, top_p is raised to the allowed floor
    assert "temperature" not in w and "top_k" not in w and "min_p" not in w
    assert w["top_p"] == 0.95
    assert w["max_tokens"] == 4000 and w["stream"] is True


def test_thinking_modes():
    base = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8000, "temperature": 0.3}
    w = anthropic().wire_body({**base, "thinking": {"enabled": False, "effort": "xhigh"}}, False)
    assert w["thinking"] == {"type": "disabled"} and w["temperature"] == 0.3
    assert "output_config" not in w
    always = ProviderOptions(quirks=frozenset({"thinking_always_on", "no_sampling"}))
    w = anthropic(options=always).wire_body({**base, "thinking": {"enabled": False}}, False)
    assert w["thinking"]["type"] == "adaptive" and w["output_config"] == {"effort": "low"}
    assert "temperature" not in w
    budget = ProviderOptions(thinking="budget", thinking_budget=2048)
    w = anthropic(options=budget).wire_body(base, False)
    assert w["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    w = anthropic(options=ProviderOptions(thinking="budget")).wire_body(base, False)
    assert "thinking" not in w  # budget models think only when asked
    w = anthropic(options=ProviderOptions(cache=False)).wire_body(
        {**base, "messages": [{"role": "system", "content": "s"}, *base["messages"]]}, False
    )
    assert "cache_control" not in json.dumps(w)


def test_constrained_protocols_are_rejected():
    with pytest.raises(LLMError, match="native tool-call protocol"):
        anthropic().wire_body({"messages": [], "grammar": "root ::= x"}, False)


def test_replay_blocks_are_sent_verbatim_and_can_be_stripped():
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
    ]
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
            "replay": {"provider": "anthropic", "model": "m", "content": blocks},
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "A"},
    ]
    p = anthropic()
    assert p.wire_body({"messages": msgs}, False)["messages"][1]["content"] == blocks
    stripped = p.wire_body({"messages": msgs}, False, strip_thinking=True)
    assert stripped["messages"][1]["content"] == blocks[1:]
    other = [{**msgs[1], "replay": {"provider": "gemini", "items": []}}]
    assert p.wire_body({"messages": [msgs[0], *other, msgs[2]]}, False)["messages"][1][
        "content"
    ] == [{"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}]


def test_usage_normalisation():
    u = normalise_usage(
        {
            "input_tokens": 12,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 5000,
            "output_tokens": 40,
        }
    )
    assert u["prompt_tokens"] == 5312 and u["completion_tokens"] == 40
    assert u["prompt_tokens_details"] == {"cached_tokens": 5000}
    assert u["cache_write_tokens"] == 300


def test_missing_key_is_a_config_error():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make_provider("anthropic", model="claude-opus-5")


# -- against the mock ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
def test_chat_parses_thinking_text_and_tool_use(mock, stream):
    s = mock([{"reasoning": "check the file", "content": "Reading.", **call("read", path="a.py")}])
    p = anthropic(s)
    body = {"messages": [{"role": "user", "content": "go"}], "tools": [READ], "max_tokens": 2000}
    c = p.chat(body, stream=stream)
    assert s.errors == []
    assert c.reasoning == "check the file" and c.content == "Reading."
    assert [(t.name, json.loads(t.arguments)) for t in c.tool_calls] == [("read", {"path": "a.py"})]
    assert c.tool_calls[0].id.startswith("toolu_mock_")
    assert c.finish_reason == "tool_calls" and c.stop_raw == "tool_use"
    assert [b["type"] for b in c.replay["content"]] == ["thinking", "text", "tool_use"]
    assert c.replay["content"][0]["signature"].startswith("mocksig.")
    assert c.prompt_tokens > 0 and c.completion_tokens > 0
    assert c.token_chunks is (not stream)


def test_prompt_caching_reported(mock):
    s = mock([final("a"), final("b")])
    p = anthropic(s)
    msgs = [{"role": "system", "content": "sys " * 50}, {"role": "user", "content": "x"}]
    first = p.chat({"messages": msgs, "max_tokens": 100})
    second = p.chat({"messages": msgs + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "y"}], "max_tokens": 100})  # fmt: skip
    assert first.cache_write_tokens > 0 and first.cached_tokens == 0
    assert second.cached_tokens > 0
    assert "cache_control" in json.dumps(s.requests[1]["system"])


def test_errors(mock):
    s = mock(
        [
            {"error": {"status": 529, "message": "Overloaded"}},
            final("after retry"),
            {"error": {"status": 429, "message": "slow", "headers": {"retry-after": "0"}}},
            final("after 429"),
        ]
    )
    p = anthropic(s, retries=1)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 100}
    assert p.chat(body).content == "after retry"
    assert p.chat(body).content == "after 429"
    with pytest.raises(ContextOverflow, match="prompt is too long"):
        anthropic(mock([final()], n_ctx=3)).chat(body)


def test_mid_stream_error_is_not_retried(mock):
    s = mock([{"content": "partial", "stream_error": {"type": "overloaded_error"}}, final("x")])
    with pytest.raises(LLMError, match="overloaded_error"):
        anthropic(s, retries=2).chat({"messages": [{"role": "user", "content": "x"}]})
    assert len(s.requests) == 1


def test_auth_headers(mock):
    s = mock([final()])
    p = make_provider(
        "anthropic", base_url=f"{s.root}/v1", model="m", api_key="", require_key=False
    )
    with pytest.raises(LLMError, match="401"):
        p.chat({"messages": [{"role": "user", "content": "x"}]})


def test_detect_and_count_tokens(mock):
    s = mock([], n_ctx=200_000)
    p = anthropic(s)
    assert p.context_window() == 200_000 and p.detect().max_output == 64000
    n = p.count_prompt({"messages": [{"role": "user", "content": "hello there"}]})
    assert n and n > 0


# -- the agent on Anthropic ------------------------------------------------------------------------


def claude(make_agent, steps, *overrides, **server_kw):
    return make_agent(
        steps,
        overrides=['model.provider="anthropic"', 'model.model="claude-opus-5"', *overrides],
        server_kw=server_kw,
    )


def test_agent_run_with_thinking_replay(make_agent, workspace, key):
    steps = [
        {"reasoning": "read it first", **call("read", path="calc.py")},
        {"reasoning": "flip the sign", **call("edit", path="calc.py", old="return a - b", new="return a + b")},
        final("Fixed add()."),
    ]  # fmt: skip
    agent, server = claude(make_agent, steps)
    result = agent.run("fix add")
    assert result.status == "completed", result.failures
    assert server.errors == []
    assert "return a + b" in (workspace / "calc.py").read_text()
    # turn 2's request replays turn 1's thinking block with its signature
    replayed = server.requests[1]["messages"][1]["content"]
    assert replayed[0]["type"] == "thinking" and replayed[0]["signature"].startswith("mocksig.")
    run = agent.store.run(result.run_id)
    assert run["provider"].startswith("anthropic") and run["cache_read_tokens"] > 0
    turns = agent.store.turns(result.run_id)
    assert turns[0]["reasoning"] == "read it first" and turns[0]["model"] == "claude-opus-5"
    msgs = agent.store.messages(result.run_id)
    assert any(m["replay_json"] and "mocksig" in m["replay_json"] for m in msgs)


def test_history_edit_strips_thinking_and_retries(make_agent, key):
    steps = [
        {"reasoning": "look", **call("read", path="calc.py")},
        {"reasoning": "again", **call("read", path="pkg/util.py")},
        {"reasoning": "done", **final("ok")},
    ]
    agent, server = claude(make_agent, steps)

    def edit_history(turn, run):
        if turn == 0:
            mid, msg = run.context[1]  # the task message
            run.context[1] = (mid, {**msg, "content": msg["content"] + " (edited)"})

    result = agent.run("fix add", after_turn=edit_history)
    assert result.status == "completed", result.failures
    notes = agent.store.events(result.run_id, "provider_note")
    assert notes and "thinking blocks stripped" in notes[0]["data_json"]
    # the rejected request, the retry without thinking blocks, and no replay afterwards
    assert len(server.requests) == 4
    assert "thinking" not in json.dumps(server.requests[2]["messages"][:3])
    assert len(notes) == 1 and agent.store.events(result.run_id, "replay_dropped")


def test_compaction_drops_replay(make_agent, workspace, key):
    (workspace / "big.txt").write_text("\n".join(f"line {i} " + "x" * 60 for i in range(60)))
    steps = [
        {"reasoning": "big file", **call("read", path="big.txt")},
        {"reasoning": "now calc", **call("read", path="calc.py")},
        {"reasoning": "and util", **call("read", path="pkg/util.py")},
        final("Read them."),
    ]
    agent, server = claude(
        make_agent,
        steps,
        "model.context_window=3000",
        "compaction.threshold=0.5",
        "compaction.keep_last_turns=1",
        "compaction.summarize=false",
    )
    result = agent.run("read the files")
    assert result.status == "completed", result.failures
    assert agent.store.compactions(result.run_id)
    assert agent.store.events(result.run_id, "replay_dropped")
    assert not agent.store.events(result.run_id, "provider_note")  # nothing was rejected
    assert server.errors == []


def test_a_refusal_ends_the_run_and_the_session_goes_on(make_agent, key):
    steps = [{"content": "Sure, here", "refusal": {"category": "cyber"}}, final("rephrased ok")]
    agent, server = claude(make_agent, steps)
    first = agent.run("something")
    assert (first.status, first.final) == ("refused", "The request was declined.")
    assert ("refusal", "The request was declined.") in first.failures
    assert agent.store.events(first.run_id, "refusal")
    sid = agent.store.run(first.run_id)["session_id"]
    second = agent.run("something else", session=sid)
    assert second.status == "completed" and server.errors == []
    # the refused reply is left out; the two user turns go as one message
    wire = server.requests[1]["messages"]
    assert [m["role"] for m in wire] == ["user"]
    assert [b["text"] for b in wire[0]["content"]] == ["something", "something else"]
    kinds = [m["role"] for m in agent.store.messages(first.run_id)]
    assert "assistant" in kinds  # still in the log


def test_max_tokens_mid_tool_call_is_malformed(make_agent, key):
    long = {"path": "calc.py", "old": "return a - b", "new": "return a + b  # " + "x " * 200}
    steps = [call("edit", **long), final("gave up")]
    agent, server = claude(make_agent, steps, "model.max_tokens=40")
    result = agent.run("fix")
    tags = [t for t, _ in result.failures]
    assert "malformed_call" in tags
    assert agent.store.turns(result.run_id)[0]["finish_reason"] == "length"


def test_overhead_uses_count_tokens(make_agent, key):
    agent, server = claude(make_agent, [])
    counter = TokenCounter(agent.llm)
    ov = measure(agent, detailed=True, counter=counter)
    assert ov.method == "api"
    assert ov.total > ov.system > 0 and ov.tools > 0
    assert set(ov.per_tool) >= {"read", "edit", "bash"}
    assert ov.context_window == server.n_ctx
