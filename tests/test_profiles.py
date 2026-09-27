"""Model profiles, protocol fallback, cost, `argus models` and `argus tune`."""

from __future__ import annotations

import tomllib

import pytest

from argus.cli import main
from argus.config import ConfigError, load_config
from argus.mock import MockServer, Script, call, final, raw_call
from argus.mock.script import request_protocol
from argus.profiles import (
    ProfileError,
    dump_profiles,
    load_profiles,
    match,
    save_user_profile,
    turn_cost,
)
from argus.providers.base import Completion

from .conftest import q


@pytest.fixture
def user_models(tmp_path, monkeypatch):
    path = tmp_path / "models.toml"
    monkeypatch.setenv("ARGUS_MODELS", str(path))
    return path


# -- catalog and matching ------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,provider,want",
    [
        ("claude-opus-5", "anthropic", "claude-opus-5"),
        ("claude-opus-5-5", "anthropic", "claude-opus-5-5"),
        ("claude-sonnet-5", "auto", "claude-sonnet-5"),
        ("claude-3-7-sonnet-latest", "anthropic", "claude"),
        ("gpt-5-mini", "openai", "gpt-5-mini"),
        ("gpt-5", "openai", "gpt-5"),
        ("gemini-3-pro-preview", "gemini", "gemini-3-pro"),
        ("qwen3-coder:30b", "ollama", "qwen3-coder"),
        ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "llama_server", "qwen3-coder"),
        ("qwen3-32b", "vllm", "qwen3"),
        ("claude-opus-5", "openrouter", None),  # an Anthropic profile is not for OpenRouter
        ("qwen", "auto", None),
    ],
)
def test_matching(model, provider, want):
    p = match(load_profiles(), model, provider)
    assert (p.name if p else None) == want


def test_user_profiles_override_and_extend(user_models):
    user_models.write_text(
        """
[profiles.claude-opus-5]
effort = "high"

[profiles.my-coder]
extends = "qwen3-coder"
match = ["my-coder-*"]
max_output = 4096
"""
    )
    profiles = load_profiles()
    opus = profiles["claude-opus-5"]
    assert (opus.effort, opus.max_output, opus.source) == ("high", 64000, "builtin+user")
    mine = profiles["my-coder"]
    assert mine.protocol == "grammar" and mine.sampling["top_k"] == 20
    assert mine.max_output == 4096 and mine.match == ["my-coder-*"]
    assert match(profiles, "my-coder-7b").name == "my-coder"


@pytest.mark.parametrize(
    "text,err",
    [
        ("[profiles.x]\nbogus = 1\n", "unknown profile key"),
        ('[profiles.x]\nextends = "y"\n', "unknown profile"),
        ('[profiles.x]\nextends = "y"\n[profiles.y]\nextends = "x"\n', "cycle"),
    ],
)
def test_bad_user_profiles(user_models, text, err):
    user_models.write_text(text)
    with pytest.raises(ProfileError, match=err):
        load_profiles()
    with pytest.raises(ConfigError):
        load_config(None, ['model.model="x"'])


# -- applying profiles ---------------------------------------------------------------------------


def test_profile_fills_only_unset_keys(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[model]\nmax_tokens = 1000\n[agent]\nprotocol = "native"\n')
    cfg = load_config(cfg_file, ['model.model="qwen3-coder-30b"', "model.temperature=0.1"])
    assert cfg.model.profile == "qwen3-coder"
    assert cfg.model.max_tokens == 1000  # from the file
    assert cfg.agent.protocol == "native"  # from the file
    assert cfg.model.temperature == 0.1  # from -o
    assert cfg.model.top_k == 20  # from the profile
    assert {"model.max_tokens", "agent.protocol", "model.temperature"} <= cfg.explicit


def test_profile_by_name_and_protocol_fallback(user_models):
    user_models.write_text(
        '[profiles.local-coder]\nextends = "qwen3-coder"\nmodel = "qwen3-coder:30b"\n'
        'provider = "ollama"\n'
    )
    cfg = load_config(None, ['model.model="local-coder"'])
    assert cfg.model.model == "qwen3-coder:30b" and cfg.model.provider == "ollama"
    assert cfg.agent.protocol == "json_schema"  # Ollama cannot take a GBNF grammar


def test_cloud_profile_sets_compaction_on_the_same_api():
    cfg = load_config(None, ['model.provider="anthropic"', 'model.model="claude-sonnet-5"'])
    c = cfg.compaction
    assert (c.provider, c.model) == ("anthropic", "claude-haiku-4-5")
    cfg = load_config(
        None,
        ['model.provider="anthropic"', 'model.model="claude-sonnet-5"', 'compaction.model="x"'],
    )
    assert cfg.compaction.model == "x" and cfg.compaction.provider == "auto"


def test_detected_context_window_beats_the_catalog(make_agent):
    agent, server = make_agent(
        [], overrides=['model.model="qwen3-coder-30b"'], server_kw={"n_ctx": 4000}
    )
    assert agent.context_window() == 4000
    agent, _ = make_agent(
        [], overrides=['model.model="qwen3-coder-30b"', "model.context_window=3000"]
    )
    assert agent.context_window() == 3000


def test_explicit_unsupported_protocol_is_an_error(make_agent):
    with pytest.raises(ConfigError, match="cannot serve the 'grammar' protocol"):
        make_agent([], overrides=['agent.protocol="grammar"'], server_kw={"flavor": "ollama"})


def test_profile_protocol_falls_back_when_the_server_cannot(make_agent):
    steps = [final("ok")]
    agent, server = make_agent(
        steps, overrides=['model.model="qwen3-coder-30b"'], server_kw={"flavor": "lmstudio"}
    )
    assert agent.cfg.agent.protocol == "json_schema"
    result = agent.run("x")
    assert result.status == "completed"
    ev = agent.store.events(result.run_id, "setup")
    assert "not available" in ev[0]["data_json"]
    assert request_protocol(server.requests[0]) == "json_schema"


# -- cost -----------------------------------------------------------------------------------------


def test_turn_cost():
    c = Completion(
        usage={
            "prompt_tokens": 10_000,
            "completion_tokens": 1_000,
            "prompt_tokens_details": {"cached_tokens": 8_000},
            "cache_write_tokens": 1_000,
        }
    )
    pricing = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}
    # 1000 fresh × 5 + 8000 × 0.5 + 1000 × 6.25 + 1000 × 25, per million
    assert turn_cost(c, pricing) == pytest.approx((5000 + 4000 + 6250 + 25000) / 1e6)
    assert turn_cost(c, {}) is None
    assert turn_cost(Completion(usage={"cost": 0.0123}), {}) == 0.0123


def test_run_cost_recorded_and_reported(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-000000000")
    agent, _ = make_agent(
        [call("read", path="calc.py"), final("ok")],
        overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"'],
    )
    result = agent.run("look")
    assert result.cost_usd and result.cost_usd > 0
    run = agent.store.run(result.run_id)
    assert run["cost_usd"] == pytest.approx(result.cost_usd, rel=1e-3)
    turns = agent.store.turns(result.run_id)
    assert all(t["cost_usd"] > 0 for t in turns)


def test_openrouter_pricing_from_detection(make_agent, monkeypatch):
    agent, _ = make_agent([final("ok")], server_kw={"flavor": "openrouter"})
    assert agent.pricing()["input"] == 0.2
    result = agent.run("x")
    assert result.cost_usd and result.cost_usd > 0  # OpenRouter's usage.cost


# -- writing profiles -----------------------------------------------------------------------------


def test_save_user_profile_round_trip(user_models):
    save_user_profile("qwen3-coder:30b", {"match": ["qwen3-coder:30b"], "protocol": "grammar"})
    save_user_profile(
        "qwen3-coder:30b", {"extends": "qwen3-coder", "tuned": {"winner": "grammar", "n": 3}}
    )
    data = tomllib.loads(user_models.read_text())
    table = data["profiles"]["qwen3-coder:30b"]
    assert table["protocol"] == "grammar" and table["tuned"] == {"winner": "grammar", "n": 3}
    assert load_profiles()["qwen3-coder:30b"].sampling["top_k"] == 20
    text = dump_profiles({"a b": {"x": 'q"uote', "y": [1, 2.5, True], "z": {"k": "v"}}})
    assert tomllib.loads(text)["profiles"]["a b"] == {
        "x": 'q"uote',
        "y": [1, 2.5, True],
        "z": {"k": "v"},
    }


# -- argus models / argus tune ----------------------------------------------------------------------


def test_models_command(capsys, mock):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5" in out and "qwen3-coder" in out
    s = mock([], flavor="vllm", n_ctx=16384)
    assert (
        main(["models", "vllm/Qwen3-Coder-30B", "--detect", "-o", f"model.base_url={q(s.url)}"])
        == 0
    )
    out = capsys.readouterr().out
    assert "profile:   qwen3-coder" in out and "context window 16384" in out
    assert "grammar" in out


def policy(req, server):
    """Works under the grammar protocol; emits broken calls under native."""
    done = any(
        m.get("role") == "tool" or "<tool_response>" in str(m.get("content"))
        for m in req["messages"]
    )
    if request_protocol(req) == "native":
        return raw_call("edit", '{"path": "calc.py", "old": ')
    if done:
        return final("Fixed.")
    return call("edit", path="calc.py", old="return a - b", new="return a + b")


def test_tune_picks_the_protocol_that_works(tmp_path, user_models, capsys):
    fixture = tmp_path / "suite" / "fx"
    fixture.mkdir(parents=True)
    (fixture / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    suite = tmp_path / "suite" / "suite.toml"
    suite.write_text(
        'name = "t"\n[[task]]\nid = "fix"\nprompt = "fix add"\nworkspace = "fx"\n'
        "check = \"grep -q 'a + b' calc.py\"\n"
        'overrides = ["tools.require_read_before_edit=false"]\n'
    )
    server = MockServer(Script.always(policy)).start()
    try:
        args = [
            "tune",
            "llama/qwen3-coder-7b",
            "--suite",
            str(suite),
            "--protocols",
            "native,grammar",
            "-n",
            "2",
            "-q",
            "-o",
            f"model.base_url={q(server.url)}",
            "-o",
            f"log.db={q(str(tmp_path / 'a.db'))}",
        ]
        assert main(args) == 0
    finally:
        server.stop()
    out = capsys.readouterr().out
    assert "→ grammar" in out and "saved protocol=grammar" in out
    table = tomllib.loads(user_models.read_text())["profiles"]["qwen3-coder-7b"]
    assert table["protocol"] == "grammar" and table["extends"] == "qwen3-coder"
    assert table["tuned"]["results"] == {"grammar": "2/2", "native": "0/2"}
    cfg = load_config(None, ['model.model="qwen3-coder-7b"', 'model.provider="llama_server"'])
    assert cfg.model.profile == "qwen3-coder-7b" and cfg.agent.protocol == "grammar"


def test_tune_dry_run_saves_nothing(tmp_path, user_models, capsys):
    server = MockServer(Script.always(policy)).start()
    try:
        rc = main(
            [
                "tune",
                "llama/qwen3-coder-7b",
                "--protocols",
                "grammar",
                "-t",
                "fix-add",
                "-n",
                "1",
                "-q",
                "--dry-run",
                "-o",
                f"model.base_url={q(server.url)}",
                "-o",
                f"log.db={q(str(tmp_path / 'a.db'))}",
                "-o",
                "tools.require_read_before_edit=false",
            ]
        )
    finally:
        server.stop()
    assert rc == 0
    assert "dry run" in capsys.readouterr().out
    assert not user_models.exists()
