"""Model aliases and choices, the live token meter, and streaming in the console."""

from __future__ import annotations

import io

from argus.choices import cloud, local, model_choices, recent
from argus.cli import model_overrides
from argus.console import ConsoleReporter
from argus.mock import final
from argus.providers.base import Completion, Delta
from argus.tui.live import Meter, constrained_view, fmt_k, sparkline

# -- aliases and specs ---------------------------------------------------------------------------


def test_aliases_and_server_urls(tmp_path, monkeypatch):
    assert model_overrides("opus") == [
        'model.provider="anthropic"',
        'model.model="claude-opus-5-5"',
    ]
    assert model_overrides("flash")[1] == 'model.model="gemini-2.5-flash"'
    assert model_overrides("ollama/qwen3-coder:30b@http://gpu:11434/v1") == [
        'model.provider="ollama"',
        'model.model="qwen3-coder:30b"',
        'model.base_url="http://gpu:11434/v1"',
    ]
    assert (
        model_overrides("opus@https://gw.example/v1")[-1]
        == 'model.base_url="https://gw.example/v1"'
    )
    assert model_overrides("name@not-a-url") == ['model.model="name@not-a-url"']
    user = tmp_path / "models.toml"
    user.write_text(
        '[aliases]\nbox = "ollama/qwen3-coder:30b@http://box:11434/v1"\nopus = "anthropic/claude-opus-5"\n'
    )
    monkeypatch.setenv("ARGUS_MODELS", str(user))
    assert model_overrides("box")[-1] == 'model.base_url="http://box:11434/v1"'
    assert model_overrides("opus")[1] == 'model.model="claude-opus-5"'  # the user's wins


# -- choices ----------------------------------------------------------------------------------------


def test_cloud_choices_follow_the_keys(monkeypatch):
    by_name = {c.name: c for c in cloud()}
    assert not by_name["opus"].ready and by_name["opus"].note == "needs ANTHROPIC_API_KEY"
    assert "local" not in by_name  # local servers are listed when they answer
    assert by_name["qwen-cloud"].spec == "openrouter/qwen/qwen3-coder"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0")
    assert {c.name: c for c in cloud()}["opus"].ready


def test_local_choices_come_from_running_servers(mock, monkeypatch):
    server = mock([], model="qwen3-coder-30b", n_ctx=65536)
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", f"llama_server={server.url}")
    (c,) = local()
    assert c.spec == f"llama_server/qwen3-coder-30b.gguf@{server.url}"
    assert c.ready and c.note == "llama_server ctx 65k"
    assert model_overrides(c.spec)[-1] == f'model.base_url="{server.url}"'


def test_recent_choices_from_the_log(make_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0")
    local_agent, server = make_agent([final("one")])
    local_agent.run("x")
    claude, _ = make_agent(
        [final("two")], overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"']
    )
    claude.llm.base_url = server.url
    claude.store = local_agent.store
    claude.run("y")
    specs = [c.spec for c in recent(local_agent.store)]
    assert specs == ["anthropic/claude-sonnet-5", f"llama_server/qwen@{server.url}"]
    everything = model_choices(local_agent.store, probe=False)
    assert [c.group for c in everything[:2]] == ["recent", "recent"]
    assert "anthropic/claude-sonnet-5" not in [c.spec for c in everything[2:]]  # no duplicates


# -- the meter and streaming views ---------------------------------------------------------------------


def test_constrained_view():
    assert constrained_view('{"thought":"read calc') == ("", "read calc")
    assert constrained_view('{"thought":"line\\nnext \\"q\\" \\u00e9","tool":"read"') == (
        "",
        'line\nnext "q" é',
    )
    assert constrained_view('{"thought":"x\\u00') == ("", "x")  # a cut-off escape waits
    assert constrained_view('{"tool":"grep","args":{') == ("", "→ grep(…)")
    assert constrained_view("<think>checking the test") == ("checking the test", "")
    assert constrained_view('<think>plan</think>\n{"thought":"go"') == ("plan", "go")
    assert constrained_view("{") == ("", "")


def test_meter_lines(monkeypatch):
    t = [100.0]
    monkeypatch.setattr("argus.tui.live.time.monotonic", lambda: t[0])
    m = Meter(window=65536)
    assert m.lines()[0] == ("○ idle", "phase-idle")
    m.start_run()
    m.start_turn(0)
    t[0] += 1
    m.stream("reasoning", 10)
    t[0] += 2
    m.stream("reasoning", 90)
    lines = [text for text, _ in m.lines()]
    assert lines[0].endswith("thinking")
    assert lines[1] == "↓ 90 tok · 45/s"
    assert lines[2] == "turn 1 · 00:03"
    m.end_turn(12000, 95)
    m.tool("bash")
    assert m.lines()[0][0].endswith("running bash")
    m.stop()
    lines = [text for text, _ in m.lines()]
    assert lines[:4] == ["○ idle", "last ↓ 95 tok · 48/s", "ctx 18% of 66k", "in 12k · out 95"]
    assert all(len(line) <= 28 for line in lines)


def test_small_formatters():
    assert [fmt_k(n) for n in (5, 1234, 65536, 2_500_000)] == ["5", "1.2k", "66k", "2.5M"]
    assert sparkline([1, 2, 3]) == "▁▅█"
    assert sparkline([4, 4]) == "▅▅"


# -- console -----------------------------------------------------------------------------------------------


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_console_live_token_line(monkeypatch):
    t = [0.0]
    monkeypatch.setattr("argus.console.time.monotonic", lambda: t[0])
    out = TTY()
    rep = ConsoleReporter(out, color=False)  # a terminal without -v: the live line
    assert rep.live and not rep.verbose
    c = Completion()
    rep.turn_start(0)
    for i in range(1, 41):
        t[0] += 0.05
        c.content_chunks = i
        rep.delta(Delta("content", "x", c))
    assert "\r\x1b[2K  writing · ↓ 39 tok · 21 tok/s" in out.getvalue()  # redrawn every 0.1 s
    rep.note("done")
    assert out.getvalue().endswith("\r\x1b[2K  · done\n")  # the live line is cleared first
    assert not ConsoleReporter(io.StringIO()).live  # not a terminal: no live line
    assert not ConsoleReporter(TTY(), verbose=True).live  # streaming shows it all anyway


def test_console_turn_stats_have_a_rate():
    out = io.StringIO()
    rep = ConsoleReporter(out, color=False)
    c = Completion(
        usage={"prompt_tokens": 100, "completion_tokens": 50}, ttft_ms=500.0, total_ms=1500.0
    )

    class P:
        calls: list = []
        content = ""

    rep.turn_end(0, c, P())
    assert "[0] 100→50 tok, 1.5s, 50 tok/s" in out.getvalue()
