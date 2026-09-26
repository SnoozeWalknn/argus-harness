from __future__ import annotations

import json

import httpx
import pytest

from argus.mock import MockServer, Script, call, final, render_chatml, tokenize
from argus.mock.script import check_expect


def post(server: MockServer, path: str, body: dict) -> httpx.Response:
    return httpx.post(server.url.removesuffix("/v1") + path, json=body, timeout=10, trust_env=False)


def chat(server: MockServer, body: dict) -> dict:
    r = httpx.post(server.url + "/chat/completions", json=body, timeout=10, trust_env=False)
    return {"status": r.status_code, **r.json()}


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]


def test_non_stream_tool_call_and_final(mock):
    s = mock([call("read", path="a.py"), final("all done")])
    r = chat(s, {"messages": MSGS, "tools": TOOLS})
    msg = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "read"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert r["usage"]["prompt_tokens"] > 0 and r["usage"]["completion_tokens"] > 0
    assert "timings" in r
    r2 = chat(s, {"messages": MSGS, "tools": TOOLS})
    assert r2["choices"][0]["message"]["content"] == "all done"
    assert r2["choices"][0]["finish_reason"] == "stop"
    assert len(s.requests) == 2


def test_script_exhausted_is_an_error(mock):
    s = mock([])
    r = chat(s, {"messages": MSGS})
    assert r["status"] == 500 and "exhausted" in r["error"]["message"]


def test_max_tokens_truncates_with_length(mock):
    s = mock([{"reasoning": "think ", "reasoning_repeat": 1000}])
    r = chat(s, {"messages": MSGS, "max_tokens": 50})
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 50


def test_truncated_tool_call_surfaces_as_content(mock):
    s = mock(
        [
            {
                "content": "x",
                "tool_calls": [{"name": "edit", "arguments": {"path": "a", "new": "y " * 100}}],
            }
        ]
    )
    r = chat(s, {"messages": MSGS, "tools": TOOLS, "max_tokens": 20})
    msg = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "length"
    assert "tool_calls" not in msg
    assert "<tool_call>" in msg["content"]


def test_context_overflow_error(mock):
    s = mock([final()], n_ctx=10)
    r = chat(s, {"messages": MSGS})
    assert r["status"] == 400
    assert r["error"]["type"] == "exceed_context_size_error"


def test_cache_reuse_reported(mock):
    s = mock([final("a"), final("b")])
    chat(s, {"messages": MSGS})
    r = chat(
        s,
        {
            "messages": MSGS
            + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}]
        },
    )
    assert r["timings"]["cache_n"] > 0
    assert r["timings"]["prompt_n"] < r["usage"]["prompt_tokens"]


def test_stream_chunks(mock):
    s = mock(
        [
            {
                "reasoning": "hmm ok",
                "content": "",
                "tool_calls": [{"name": "read", "arguments": {"path": "x"}}],
            }
        ]
    )
    events = []
    with httpx.stream(
        "POST",
        s.url + "/chat/completions",
        json={
            "messages": MSGS,
            "tools": TOOLS,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        trust_env=False,
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    reasoning = "".join(
        e["choices"][0]["delta"].get("reasoning_content") or "" for e in events if e["choices"]
    )
    assert reasoning == "hmm ok"
    args = "".join(
        tc["function"].get("arguments", "")
        for e in events
        if e["choices"]
        for tc in e["choices"][0]["delta"].get("tool_calls") or []
    )
    assert json.loads(args) == {"path": "x"}
    assert events[-1]["usage"]["completion_tokens"] > 0


def test_tokenize_and_apply_template(mock):
    s = mock([])
    toks = post(s, "/tokenize", {"content": "hello world"}).json()["tokens"]
    assert len(toks) == len(tokenize("hello world")) == 3
    prompt = post(s, "/apply-template", {"messages": MSGS, "tools": TOOLS}).json()["prompt"]
    assert "<tools>" in prompt and prompt.endswith("<|im_start|>assistant\n")
    assert (
        render_chatml({"messages": MSGS})
        in post(s, "/apply-template", {"messages": MSGS}).json()["prompt"]
    )


def test_props_and_health(mock):
    s = mock([], n_ctx=4096)
    root = s.url.removesuffix("/v1")
    assert (
        httpx.get(root + "/props", trust_env=False).json()["default_generation_settings"]["n_ctx"]
        == 4096
    )
    assert httpx.get(root + "/health", trust_env=False).status_code == 200


def test_expectations(mock):
    s = mock([{"expect": {"last_contains": "nope"}, "content": "x"}])
    r = chat(s, {"messages": MSGS})
    assert r["status"] == 500
    assert s.errors and "lacks 'nope'" in s.errors[0]


def test_check_expect_keys():
    req = {"messages": MSGS, "tools": TOOLS, "max_tokens": 5}
    assert (
        check_expect(
            {"last_role": "user", "tools": ["read"], "max_tokens": 5, "has": ["tools"]}, req
        )
        == []
    )
    assert check_expect({"protocol": "grammar"}, req)
    assert check_expect({"bogus": 1}, req)


def test_callable_step_and_times(mock):
    seen = []

    def step(req, server):
        seen.append(len(req["messages"]))
        return final(f"n={len(req['messages'])}")

    s = mock(Script([{"times": 2, "content": "x"}, step]))
    assert chat(s, {"messages": MSGS})["choices"][0]["message"]["content"] == "x"
    assert chat(s, {"messages": MSGS})["choices"][0]["message"]["content"] == "x"
    assert chat(s, {"messages": MSGS})["choices"][0]["message"]["content"] == "n=2"
    assert seen == [2]


def test_http_error_step(mock):
    s = mock([{"error": {"status": 503, "message": "loading model"}}])
    r = chat(s, {"messages": MSGS})
    assert r["status"] == 503


def test_script_load(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"steps": [final("a")], "on_exhausted": "final"}))
    sc = Script.load(p)
    assert sc.next({}, None)["content"] == "a"
    assert sc.next({}, None)["content"] == "Done."
    with pytest.raises(ValueError):
        Script([], on_exhausted="bogus")
