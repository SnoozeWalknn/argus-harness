from __future__ import annotations

import pytest

from argus.llm import ContextOverflow, LLMClient, LLMError, StopGeneration, root_url
from argus.mock import call, final

MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]


@pytest.mark.parametrize("stream", [True, False])
def test_chat_assembles_tool_calls(mock, stream):
    s = mock(
        [
            {
                "reasoning": "let me look",
                "content": "Checking.",
                "tool_calls": [
                    {"name": "read", "arguments": {"path": "a.py"}},
                    {"name": "grep", "arguments": {"pattern": "def "}},
                ],
            },
        ]
    )
    c = LLMClient(s.url, retries=0).chat({"messages": MSGS, "tools": TOOLS}, stream=stream)
    assert c.reasoning == "let me look"
    assert c.content == "Checking."
    assert [(t.name, t.arguments) for t in c.tool_calls] == [
        ("read", '{"path": "a.py"}'),
        ("grep", '{"pattern": "def "}'),
    ]
    assert c.finish_reason == "tool_calls"
    assert c.prompt_tokens > 0 and c.completion_tokens > 0
    assert c.timings["predicted_n"] == c.completion_tokens
    if stream:
        assert c.ttft_ms is not None
        assert c.reasoning_chunks == 5  # "let", " ", "me", " ", "look"


def test_monitor_aborts_stream(mock):
    s = mock([{"reasoning": "again ", "reasoning_repeat": 5000, "chunk_delay": 0.0005}])
    seen = []

    def monitor(d):
        seen.append(d.text)
        if d.completion.reasoning_chunks >= 30:
            raise StopGeneration("reasoning_budget", "30 tokens")

    c = LLMClient(s.url, retries=0).chat({"messages": MSGS}, stream=True, monitors=[monitor])
    assert c.aborted == "reasoning_budget"
    assert c.reasoning_chunks == 30
    assert c.completion_tokens == 30  # estimated from chunks: no usage on an aborted stream


def test_context_overflow_raises(mock):
    s = mock([final()], n_ctx=5)
    with pytest.raises(ContextOverflow):
        LLMClient(s.url, retries=0).chat({"messages": MSGS}, stream=True)
    with pytest.raises(ContextOverflow):
        LLMClient(s.url, retries=0).chat({"messages": MSGS}, stream=False)


def test_http_error_raises(mock):
    s = mock([{"error": {"status": 500, "message": "boom"}}])
    with pytest.raises(LLMError, match="boom"):
        LLMClient(s.url, retries=0).chat({"messages": MSGS})


def test_unreachable_server():
    with pytest.raises(LLMError, match="cannot reach"):
        LLMClient("http://127.0.0.1:9/v1", retries=0, connect_timeout=0.5).chat({"messages": MSGS})


def test_native_endpoints(mock):
    s = mock([call("read", path="x")], n_ctx=1234)
    c = LLMClient(s.url)
    assert c.context_window() == 1234
    assert len(c.tokenize("a b")) == 3
    assert "<|im_start|>user" in c.apply_template({"messages": MSGS})
    assert c.health()


def test_root_url():
    assert root_url("http://h:1/v1") == "http://h:1"
    assert root_url("http://h:1/v1/") == "http://h:1"
    assert root_url("http://h:1") == "http://h:1"


def test_read_timeout_becomes_llm_error(mock):
    s = mock([{"delay": 2, "content": "late"}, {"content": "x", "chunk_delay": 2}])
    client = LLMClient(s.url, retries=0, timeout=0.5)
    with pytest.raises(LLMError, match="timed out"):
        client.chat({"messages": MSGS}, stream=False)
    with pytest.raises(LLMError, match="stalled|timed out"):
        client.chat({"messages": MSGS}, stream=True)
