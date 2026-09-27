"""Provider layer: model specs, keys, OpenAI-compatible flavours, quirks, retries, recording."""

from __future__ import annotations

import json

import pytest

from argus.config import ConfigError
from argus.mock import MockServer, Script, call, final
from argus.providers import (
    ProviderOptions,
    api_key_for,
    make_provider,
    parse_spec,
    resolve_kind,
)
from argus.providers.base import LLMError, StopGeneration, iter_sse
from argus.providers.record import Recorder

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def provider(server: MockServer, **kw):
    kind = kw.pop("kind", "openai_compat")
    return make_provider(kind, base_url=server.url, model=server.model, retries=0, **kw)


# -- specs and keys ------------------------------------------------------------------------


def test_parse_spec():
    assert parse_spec("anthropic/claude-opus-5") == ("anthropic", "claude-opus-5")
    assert parse_spec("openrouter/qwen/qwen3-coder") == ("openrouter", "qwen/qwen3-coder")
    assert parse_spec("ollama/qwen3-coder:30b") == ("ollama", "qwen3-coder:30b")
    assert parse_spec("llama/qwen") == ("llama_server", "qwen")
    assert parse_spec("llama") == ("llama_server", "")
    assert parse_spec("Qwen/Qwen3-Coder") == (None, "Qwen/Qwen3-Coder")
    assert parse_spec("gpt-5") == (None, "gpt-5")


def test_resolve_kind_by_host():
    assert resolve_kind("auto", "https://api.anthropic.com/v1") == "anthropic"
    assert resolve_kind("auto", "https://api.openai.com/v1") == "openai"
    assert resolve_kind("auto", "https://generativelanguage.googleapis.com/v1beta") == "gemini"
    assert resolve_kind("auto", "https://openrouter.ai/api/v1") == "openrouter"
    assert resolve_kind("auto", "http://127.0.0.1:8080/v1") == "openai_compat"
    assert resolve_kind("auto", "") == "openai_compat"
    assert resolve_kind("ollama", "https://api.anthropic.com") == "ollama"
    with pytest.raises(ValueError):
        resolve_kind("nope")


def test_api_keys_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert api_key_for("gemini") == "g-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    assert api_key_for("gemini") == "gem-key"
    monkeypatch.setenv("MY_KEY", "mine")
    assert api_key_for("gemini", env_name="MY_KEY") == "mine"
    assert api_key_for("gemini", explicit="x") == "x"
    assert api_key_for("openai_compat") == ""


def test_cloud_provider_without_key_is_a_config_error():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        make_provider("openrouter", model="m")
    with pytest.raises(ConfigError, match="MY_VAR"):
        make_provider("openrouter", model="m", api_key_env="MY_VAR")


def test_openrouter_key_sent_as_bearer(mock, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    s = mock([final("ok")], flavor="openrouter")
    p = make_provider("openrouter", base_url=s.url, model=s.model, retries=0)
    assert p.http.headers["authorization"] == "Bearer or-secret"
    assert p.chat({"messages": MSGS}).content == "ok"


# -- flavour detection -----------------------------------------------------------------------


def test_detect_llama_server(mock):
    s = mock([], n_ctx=4321)
    d = provider(s).detect()
    assert d.flavor == "llama_server" and d.context_window == 4321
    assert {"grammar", "tools", "thinking"} <= d.capabilities


def test_detect_ollama_without_num_ctx_warns(mock):
    s = mock([], flavor="ollama", n_ctx=40960)
    p = provider(s)
    d = p.detect()
    assert d.flavor == "ollama"
    assert d.context_window == 4096  # what Ollama actually serves by default
    assert d.raw["trained_context"] == 40960
    assert "num_ctx" in d.notes[0]
    assert "thinking" in d.capabilities


def test_detect_ollama_with_num_ctx(mock):
    s = mock([], flavor="ollama", n_ctx=40960, num_ctx=32768)
    assert provider(s).context_window() == 32768


def test_detect_vllm_lmstudio_generic(mock):
    assert provider(mock([], flavor="vllm", n_ctx=16384)).detect().flavor == "vllm"
    assert provider(mock([], flavor="vllm", n_ctx=16384)).context_window() == 16384
    d = provider(mock([], flavor="lmstudio", n_ctx=8192)).detect()
    assert (d.flavor, d.context_window) == ("lmstudio", 8192)
    d = provider(mock([], flavor="generic")).detect()
    assert (d.flavor, d.context_window) == ("generic", 0)


def test_detect_openrouter_pricing(mock):
    s = mock([], flavor="openrouter", n_ctx=262144)
    d = provider(s, kind="openrouter", api_key="k").detect()
    assert d.flavor == "openrouter" and d.context_window == 262144
    assert d.max_output == 16384
    assert d.pricing == {"input": 0.2, "output": 0.8, "cache_read": 0.05, "cache_write": 0.0}


def test_unreachable_server_detection_is_retried():
    p = make_provider("openai_compat", base_url="http://127.0.0.1:9/v1", connect_timeout=0.3)
    d = p.detect()
    assert d.context_window == 0 and "detection failed" in d.notes[0]
    assert p._detected is None  # not cached: the server may come up


# -- request translation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flavor,expect",
    [
        ("llama_server", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("vllm", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("ollama", {"think": False}),
        ("openrouter", {"reasoning": {"enabled": False}}),
    ],
)
def test_thinking_off_per_flavour(mock, flavor, expect):
    s = mock([final()], flavor=flavor)
    p = provider(s, api_key="k")
    p.chat({"messages": MSGS, "thinking": {"enabled": False}})
    req = s.requests[0]
    for key, value in expect.items():
        assert req[key] == value
    assert "thinking" not in req


def test_effort_per_flavour(mock):
    s = mock([final(), final()], flavor="openrouter")
    provider(s, api_key="k").chat({"messages": MSGS, "thinking": {"effort": "high"}})
    assert s.requests[0]["reasoning"] == {"effort": "high"}
    s = mock([final()])
    provider(s).chat({"messages": MSGS, "thinking": {"enabled": True, "effort": "low"}})
    assert s.requests[0]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }


def test_quirks(mock):
    s = mock([final()], flavor="generic")
    opts = ProviderOptions(
        quirks=frozenset({"no_sampling", "max_completion_tokens", "no_system_role"})
    )
    p = provider(s, options=opts)
    p.chat({"messages": MSGS, "temperature": 0.7, "top_k": 20, "max_tokens": 99})
    req = s.requests[0]
    assert "temperature" not in req and "top_k" not in req
    assert req["max_completion_tokens"] == 99 and "max_tokens" not in req
    assert req["messages"] == [{"role": "user", "content": "sys\n\nhi"}]


def test_merge_same_role_and_replay_stripped(mock):
    s = mock([final()])
    p = provider(s, options=ProviderOptions(quirks=frozenset({"merge_same_role"})))
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "x", "replay": {"provider": "anthropic", "items": []}},
    ]
    p.chat({"messages": msgs})
    assert s.requests[0]["messages"] == [
        {"role": "user", "content": "a\n\nb"},
        {"role": "assistant", "content": "x"},
    ]


def test_grammar_becomes_guided_grammar_on_vllm(mock):
    s = mock([{"raw": '{"tool":"done","args":{"summary":"x"}}'}], flavor="vllm", strict=False)
    provider(s).chat({"messages": MSGS, "grammar": 'root ::= "x"'})
    assert s.requests[0]["guided_grammar"] == 'root ::= "x"' and "grammar" not in s.requests[0]


# -- responses -------------------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["ollama", "openrouter"])
@pytest.mark.parametrize("stream", [True, False])
def test_reasoning_field(mock, flavor, stream):
    s = mock([{"reasoning": "think hard", **call("read", path="a.py")}], flavor=flavor)
    c = provider(s, api_key="k").chat({"messages": MSGS}, stream=stream)
    assert c.reasoning == "think hard"
    assert [(t.name, t.arguments) for t in c.tool_calls] == [("read", '{"path": "a.py"}')]
    assert c.finish_reason == "tool_calls"


def test_openrouter_stream_counts_tokens_by_estimate(mock):
    s = mock([{"reasoning": "word " * 200, "content": "ok"}], flavor="openrouter")
    c = provider(s, api_key="k").chat({"messages": MSGS})
    assert not c.token_chunks
    assert c.reasoning_tokens == pytest.approx(len(c.reasoning) / 3.7, abs=1)
    assert c.usage["cost"] > 0


def test_reasoning_budget_uses_token_estimate(mock):
    from argus.detect import ReasoningBudget

    s = mock([{"reasoning": "word " * 400, "content": "ok"}], flavor="openrouter")
    c = provider(s, api_key="k").chat({"messages": MSGS}, monitors=[ReasoningBudget(50)])
    assert c.aborted == "reasoning_budget"
    assert 50 <= c.reasoning_tokens <= 60


def test_retry_after_429(mock):
    s = mock(
        [
            {"error": {"status": 429, "message": "slow down", "headers": {"retry-after": "0"}}},
            final("ok"),
        ]
    )
    p = make_provider("openai_compat", base_url=s.url, retries=1)
    assert p.chat({"messages": MSGS}).content == "ok"
    assert len(s.requests) == 2


def test_500_is_not_retried(mock):
    s = mock([{"error": {"status": 500, "message": "boom"}}, final("ok")])
    p = make_provider("openai_compat", base_url=s.url, retries=3)
    with pytest.raises(LLMError, match="boom"):
        p.chat({"messages": MSGS})
    assert len(s.requests) == 1


def test_openai_style_overflow_is_context_overflow(mock):
    from argus.providers.base import ContextOverflow

    s = mock([final()], flavor="vllm", n_ctx=5)
    with pytest.raises(ContextOverflow, match="maximum context length"):
        provider(s).chat({"messages": MSGS})


# -- SSE -------------------------------------------------------------------------------------------


def test_iter_sse():
    lines = [
        ": OPENROUTER PROCESSING",
        "",
        "event: message_start",
        'data: {"a": 1}',
        "",
        "data: line1",
        "data: line2",
        "",
        'error: {"message": "boom"}',
        "data: [DONE]",
    ]
    events = list(iter_sse(lines))
    assert [(e.event, e.data) for e in events] == [
        ("message_start", '{"a": 1}'),
        (None, "line1\nline2"),
        ("error", '{"message": "boom"}'),
        (None, "[DONE]"),
    ]


# -- recording --------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
def test_record_and_replay_round_trip(tmp_path, mock, stream):
    s = mock([{"reasoning": "hm", "content": "Reading.", **call("read", path="calc.py")}])
    rec = Recorder(tmp_path / "rec")
    p = make_provider(
        "openai_compat", base_url=s.url, api_key="sk-secret-value-123", retries=0, recorder=rec
    )
    body = {"model": "m", "messages": MSGS, "thinking": {"enabled": False}}
    c = p.chat(body, stream=stream)
    (path,) = rec.saved
    text = path.read_text()
    assert "sk-secret-value-123" not in text
    fx = json.loads(text)
    assert fx["canonical"] == body
    assert fx["request"]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert fx["flavor"] == "llama_server"
    assert ("sse" in fx["response"]) == stream
    assert fx["expect"]["tool_calls"] == [[c.tool_calls[0].id, "read", '{"path": "calc.py"}']]

    # Replaying the recording through a fresh server gives the same parse and wire request.
    replay = MockServer(Script([{"replay": fx["response"]}])).start()
    try:
        p2 = make_provider(
            "openai_compat",
            base_url=replay.url,
            retries=0,
            options=ProviderOptions(flavor=fx["flavor"]),
        )
        c2 = p2.chat(fx["canonical"], stream=fx["stream"])
        assert replay.requests[0] == fx["request"]["body"]
        assert (c2.content, c2.reasoning, c2.finish_reason) == (
            c.content,
            c.reasoning,
            "tool_calls",
        )
    finally:
        replay.stop()


def test_recording_captures_errors(tmp_path, mock):
    s = mock([{"error": {"status": 400, "message": "bad request"}}])
    rec = Recorder(tmp_path)
    p = make_provider("openai_compat", base_url=s.url, retries=0, recorder=rec)
    with pytest.raises(LLMError):
        p.chat({"messages": MSGS})
    fx = json.loads(rec.saved[0].read_text())
    assert fx["response"]["status"] == 400 and fx["error"]["type"] == "LLMError"


def test_recording_on_monitor_abort(tmp_path, mock):
    s = mock([{"reasoning": "again ", "reasoning_repeat": 500}])
    rec = Recorder(tmp_path)
    p = make_provider("openai_compat", base_url=s.url, retries=0, recorder=rec)

    def stop(d):
        if d.completion.reasoning_chunks >= 10:
            raise StopGeneration("reasoning_budget")

    c = p.chat({"messages": MSGS}, monitors=[stop])
    assert c.aborted == "reasoning_budget"
    fx = json.loads(rec.saved[0].read_text())
    assert fx["expect"]["aborted"] == "reasoning_budget"
    assert len(fx["response"]["sse"]) < 50


# -- the agent on other flavours --------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["ollama", "vllm", "lmstudio", "openrouter"])
def test_agent_runs_on_flavour(make_agent, workspace, flavor, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    steps = [
        {"reasoning": "look first", **call("read", path="calc.py")},
        call("edit", path="calc.py", old="return a - b", new="return a + b"),
        final("Fixed."),
    ]
    agent, server = make_agent(steps, server_kw={"flavor": flavor})
    result = agent.run("fix add")
    assert result.status == "completed", result.failures
    assert "return a + b" in (workspace / "calc.py").read_text()
    run = agent.store.run(result.run_id)
    assert run["provider"].startswith(flavor)
    turn = agent.store.turns(result.run_id)[0]
    assert turn["reasoning"] == "look first" and turn["provider"] == "openai_compat"


def test_stored_config_is_redacted(make_agent):
    agent, _ = make_agent([final()], overrides=['model.api_key="sk-live-abcdef"'])
    result = agent.run("x")
    run = agent.store.run(result.run_id)
    assert "sk-live-abcdef" not in run["config_json"]
    assert json.loads(run["config_json"])["model"]["api_key"] == "<redacted>"


# -- CLI and storage --------------------------------------------------------------------------------


def test_cli_model_spec_sets_provider_and_model():
    from argus.cli import _config, build_parser

    args = build_parser().parse_args(["run", "x", "-m", "ollama/qwen3-coder:30b"])
    cfg = _config(args)
    assert (cfg.model.provider, cfg.model.model) == ("ollama", "qwen3-coder:30b")
    args = build_parser().parse_args(
        ["run", "x", "-m", "anthropic/claude-opus-5", "-o", 'model.model="other"', "--record", "d"]
    )
    cfg = _config(args)
    assert (cfg.model.provider, cfg.model.model, cfg.model.record_dir) == (
        "anthropic",
        "other",
        "d",
    )


def test_store_migrates_v1_database(tmp_path):
    import sqlite3

    from argus.store import SCHEMA, Store

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.execute("PRAGMA user_version=1")
    db.execute("INSERT INTO runs (id, task, started_at) VALUES ('r1', 't', 0)")
    db.commit()
    db.close()
    s = Store(path)
    assert s.q("PRAGMA user_version")[0][0] >= 2
    cols = {r["name"] for r in s.q("PRAGMA table_info(messages)")}
    assert "replay_json" in cols
    assert s.run("r1")["task"] == "t"
    s.add_message("r1", 0, {"role": "assistant", "content": "x", "replay": {"provider": "p"}})
    assert json.loads(s.messages("r1")[0]["replay_json"]) == {"provider": "p"}
    s.close()
    Store(path).close()  # reopening a migrated database is a no-op
