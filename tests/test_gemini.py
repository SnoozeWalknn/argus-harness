"""Gemini adapter, against the mock server's /v1beta/models/*:generateContent."""

from __future__ import annotations

import json

import pytest

from argus.mock import call, calls, final
from argus.providers import ProviderOptions, make_provider
from argus.providers.base import ContextOverflow, LLMError
from argus.providers.gemini import LOCAL_ID, PLACEHOLDER_SIGNATURE, GeminiProvider, sanitize_schema

KEY = "AIza-test-0000"
READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Show a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "mode": {"const": "text"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def gemini(server=None, model="gemini-2.5-pro", **kw) -> GeminiProvider:
    base = f"{server.root}/v1beta" if server else ""
    kw.setdefault("retries", 0)
    return make_provider("gemini", base_url=base, model=model, api_key=KEY, **kw)


def test_sanitize_schema():
    assert sanitize_schema(READ["function"]["parameters"]) == {
        "type": "object",
        "properties": {"path": {"type": "string"}, "mode": {"enum": ["text"]}},
        "required": ["path"],
    }


def test_wire_body_translation():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "system", "content": "You are argus."},
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}},
                    {"id": f"{LOCAL_ID}7", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "A"},
            {"role": "tool", "tool_call_id": f"{LOCAL_ID}7", "content": "B"},
            {"role": "user", "content": "[argus: loop]"},
        ],
        "tools": [READ],
        "tool_choice": "auto",
        "max_tokens": 2048,
        "temperature": 0.4,
        "thinking": {"effort": "high"},
    }  # fmt: skip
    w = gemini().wire_body(body)
    assert w["systemInstruction"] == {"parts": [{"text": "You are argus."}]}
    assert w["contents"][0] == {"role": "user", "parts": [{"text": "fix it"}]}
    assert w["contents"][1] == {
        "role": "model",
        "parts": [
            {"functionCall": {"name": "read", "args": {"path": "a"}, "id": "toolu_1"}},
            {"functionCall": {"name": "grep", "args": {"pattern": "x"}}},  # argus id stays local
        ],
    }
    assert w["contents"][2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "read", "response": {"result": "A"}, "id": "toolu_1"}},
            {"functionResponse": {"name": "grep", "response": {"result": "B"}}},
            {"text": "[argus: loop]"},
        ],
    }
    decl = w["tools"][0]["functionDeclarations"][0]
    assert "additionalProperties" not in json.dumps(decl)
    assert w["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    gen = w["generationConfig"]
    assert gen["maxOutputTokens"] == 2048 and gen["temperature"] == 0.4
    assert gen["thinkingConfig"] == {"thinkingBudget": 24576, "includeThoughts": True}


def test_thinking_config_modes():
    body = {"messages": [{"role": "user", "content": "x"}]}
    cfg = lambda p, t: p.wire_body({**body, "thinking": t})["generationConfig"]["thinkingConfig"]  # noqa: E731
    assert cfg(gemini(), {}) == {"thinkingBudget": -1, "includeThoughts": True}
    assert cfg(gemini(), {"enabled": False}) == {"thinkingBudget": 0}
    assert cfg(gemini(), {"budget": 2000}) == {"thinkingBudget": 2000, "includeThoughts": True}
    level = gemini(options=ProviderOptions(thinking="gemini_level"))
    assert cfg(level, {"effort": "low"}) == {"thinkingLevel": "low", "includeThoughts": True}
    assert cfg(level, {}) == {"thinkingLevel": "high", "includeThoughts": True}


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("ids", [False, True])
def test_chat_thoughts_calls_and_signatures(mock, stream, ids):
    s = mock(
        [{"reasoning": "compare both", "content": "Reading.", **calls(("read", {"path": "a"}), ("read", {"path": "b"}))}],
        gemini_ids=ids,
    )  # fmt: skip
    c = gemini(s).chat(
        {"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "go"}], "tools": [READ], "max_tokens": 900},
        stream=stream,
    )  # fmt: skip
    assert s.errors == []
    assert c.reasoning == "compare both" and c.content == "Reading."
    assert [json.loads(t.arguments) for t in c.tool_calls] == [{"path": "a"}, {"path": "b"}]
    if ids:
        assert all(t.id.startswith("gcall_") for t in c.tool_calls)
    else:
        assert all(t.id.startswith(LOCAL_ID) for t in c.tool_calls)
    assert c.finish_reason == "tool_calls"
    parts = c.replay["parts"]
    assert parts[0] == {"text": "compare both", "thought": True}
    assert parts[2]["thoughtSignature"].startswith("mockgsig.gemini-2.5-pro.")
    assert "thoughtSignature" not in parts[3]
    assert c.reported_reasoning_tokens == 3 and c.completion_tokens > 3


def run_agent(make_agent, steps, model, *overrides):
    return make_agent(
        steps,
        overrides=['model.provider="gemini"', f'model.model="{model}"', *overrides],
    )


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3-pro-preview"])
def test_agent_run(make_agent, workspace, monkeypatch, model):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    steps = [
        {"reasoning": "read first", **call("read", path="calc.py")},
        {
            "reasoning": "fix",
            **call("edit", path="calc.py", old="return a - b", new="return a + b"),
        },
        final("Fixed."),
    ]
    agent, server = run_agent(make_agent, steps, model)
    result = agent.run("fix add")
    assert result.status == "completed", result.failures
    assert server.errors == []
    assert "return a + b" in (workspace / "calc.py").read_text()
    turn2 = server.requests[1]["contents"]
    assert turn2[1]["role"] == "model" and "thoughtSignature" in json.dumps(turn2[1])


def test_foreign_history_needs_placeholder_signature(mock):
    s = mock([final("ok"), final("ok")])
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_9", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            "replay": {"provider": "anthropic", "content": []},
        },
        {"role": "tool", "tool_call_id": "toolu_9", "content": "A"},
    ]  # fmt: skip
    body = {"model": "gemini-3-pro-preview", "messages": msgs, "tools": [READ]}
    with pytest.raises(LLMError, match="thought_signature"):
        gemini(s, model="gemini-3-pro-preview").chat(body)
    p = gemini(
        s,
        model="gemini-3-pro-preview",
        options=ProviderOptions(quirks=frozenset({"thought_signatures"})),
    )
    assert p.chat(body).content == "ok"
    sent = s.requests[-1]["contents"][1]["parts"][0]
    assert sent["thoughtSignature"] == PLACEHOLDER_SIGNATURE


def test_malformed_function_call_is_tagged(make_agent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    steps = [{"stop_reason": "MALFORMED_FUNCTION_CALL"}, final("ok")]
    agent, server = run_agent(make_agent, steps, "gemini-2.5-flash")
    result = agent.run("x")
    assert result.status == "completed"
    assert any(t == "malformed_call" and "malformed function call" in d for t, d in result.failures)


def test_safety_is_a_refusal(mock):
    c = gemini(mock([{"content": "x", "refusal": True}])).chat(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    assert c.finish_reason == "refusal" and c.refusal == "SAFETY"


def test_errors_count_and_detect(mock):
    s = mock(
        [{"error": {"status": 503, "message": "The model is overloaded."}}, final("ok")],
        n_ctx=1_048_576,
    )
    p = gemini(s, retries=1)
    assert p.chat({"messages": [{"role": "user", "content": "x"}]}).content == "ok"
    assert p.context_window() == 1_048_576 and p.detect().max_output == 65536
    assert p.count_prompt({"messages": [{"role": "user", "content": "count me"}]}) > 0
    with pytest.raises(ContextOverflow, match="input token count"):
        gemini(mock([final()], n_ctx=2)).chat({"messages": [{"role": "user", "content": "x y z"}]})
    with pytest.raises(LLMError, match="403"):
        make_provider("gemini", base_url=f"{s.root}/v1beta", model="m", require_key=False).chat(
            {"messages": [{"role": "user", "content": "x"}]}
        )
