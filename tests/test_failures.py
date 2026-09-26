"""Each failure tag reproduced by a mock scenario (milestone 3)."""

from __future__ import annotations

import json

import pytest

from argus.cli import main
from argus.detect import (
    LoopDetector,
    claims_completion,
    cut_at_role_marker,
    periodic_tail,
    repeated_paragraph,
)
from argus.mock import call, final


def tags(result):
    return [t for t, _ in result.failures]


# -- loop ------------------------------------------------------------------------------------------


def test_loop_same_call_same_result_nudges(make_agent):
    agent, server = make_agent(
        [call("glob", pattern="*.py")] * 3
        + [{"expect": {"last_contains": "[argus: same call and result 3x in a row"}, **final("ok")}]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["loop"]
    assert "glob(pattern=*.py)" in r.failures[0][1]


def test_loop_aborts(make_agent):
    agent, _ = make_agent([call("glob", pattern="*.py")] * 5, ["agent.loop_abort=5"])
    r = agent.run("x")
    assert r.status == "failed"
    assert tags(r) == ["loop", "loop"]
    assert r.failures[-1][1].endswith("aborting")
    assert r.turns == 5


def test_repeated_call_with_changing_result_is_not_a_loop(make_agent):
    agent, _ = make_agent([call("bash", cmd="date +%s%N")] * 4 + [final()])
    r = agent.run("x")
    assert r.status == "completed" and r.failures == []


def test_loop_cycle(make_agent):
    agent, _ = make_agent(
        [call("glob", pattern="*.py"), call("grep", pattern="def")] * 3 + [final()],
        ["agent.loop_abort=9"],
    )
    r = agent.run("x")
    assert tags(r) == ["loop"]
    assert "cycle of 2 calls repeated 3x" in r.failures[0][1]


def test_degenerate_repetition_aborts_stream_and_retries(make_agent):
    agent, server = make_agent(
        [
            {
                "reasoning": "I should check the file again. ",
                "reasoning_repeat": 2000,
                "chunk_delay": 0.0002,
            },
            {"expect": {"chat_template_kwargs": {"enable_thinking": False}}, **final("ok")},
        ]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["loop"]
    assert "reasoning degenerated into repeating" in r.failures[0][1]
    turns = agent.store.turns(r.run_id)
    assert [(t["idx"], t["attempt"], t["aborted"]) for t in turns] == [
        (0, 0, "repetition"),
        (0, 1, None),
    ]
    assert turns[0]["completion_tokens"] < 2000 * 12  # stopped early
    assert server.requests[1]["messages"][-1]["content"].startswith("Your previous reply ran away")


# -- token cap -------------------------------------------------------------------------------------


def test_reasoning_budget_retries_without_thinking(make_agent):
    agent, server = make_agent(
        [
            {"reasoning": "hmm, maybe ", "reasoning_repeat": 5000, "chunk_delay": 0.0002},
            {
                "expect": {"chat_template_kwargs": {"enable_thinking": False}},
                **call("glob", pattern="*.md"),
            },
            {"expect": {"lacks": ["chat_template_kwargs"]}, **final("ok")},
        ],
        ["agent.max_reasoning_tokens=50", "agent.repetition_window=0"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["token_cap"]
    assert "reasoning exceeded 50 tokens" in r.failures[0][1]
    t0 = agent.store.turns(r.run_id)[0]
    assert t0["aborted"] == "reasoning_budget" and t0["reasoning_tokens"] == 51


def test_runaway_reasoning_without_streaming(make_agent):
    agent, server = make_agent(
        [
            {"reasoning": "let me think ", "reasoning_repeat": 5000},
            {"expect": {"chat_template_kwargs": {"enable_thinking": False}}, **final("ok")},
        ],
        ["model.stream=false", "model.max_tokens=100", "agent.repetition_window=200"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["token_cap", "loop"]  # hit max_tokens, and the text was degenerate
    assert "without acting" in r.failures[0][1]


def test_runaway_twice_fails_the_run(make_agent):
    agent, _ = make_agent(
        [{"reasoning": "zzz ", "reasoning_repeat": 5000}] * 2,
        ["model.max_tokens=60", "agent.repetition_window=0"],
    )
    r = agent.run("x")
    assert r.status == "failed"
    assert tags(r) == ["token_cap", "token_cap"]


def test_cut_off_tool_call_gets_feedback(make_agent):
    agent, server = make_agent(
        [
            call("edit", path="calc.py", old="x", new="y " * 500),
            {"expect": {"last_contains": "cut off at the output token limit"}, **final("ok")},
        ],
        ["model.max_tokens=80"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["token_cap"]


def test_run_token_budget(make_agent):
    agent, server = make_agent(
        [call("glob", pattern="*.py")] * 5 + [final()], ["agent.token_budget=30"]
    )
    r = agent.run("x")
    assert r.status == "failed"
    assert tags(r)[-1] == "token_cap" and "budget exhausted (30)" in r.failures[-1][1]
    assert r.completion_tokens >= 30
    # the last request asked for no more than what was left of the budget
    assert server.requests[-1]["max_tokens"] <= 30 - sum(
        x["usage"]["completion_tokens"] for x in server.responses[:-1]
    )


# -- overrun -------------------------------------------------------------------------------------


def test_overrun_role_marker_trimmed(make_agent):
    agent, _ = make_agent([final("Fixed it.\n<|im_start|>user\nthanks!<|im_end|>")])
    r = agent.run("x")
    assert r.status == "completed"
    assert r.final == "Fixed it."
    assert tags(r) == ["overrun"] and "<|im_start|>" in r.failures[0][1]


def test_overrun_repeated_summary(make_agent):
    para = "I changed add() to return the sum and verified it with the tests."
    agent, _ = make_agent([final(f"{para}\n\n{para}\n\n{para}")])
    r = agent.run("x")
    assert tags(r) == ["overrun"] and "repeats itself" in r.failures[0][1]


def test_overrun_after_declared_completion(make_agent):
    agent, _ = make_agent(
        [
            {
                "content": "The fix is complete. Let me double check.",
                **call("read", path="calc.py"),
            },
            call("grep", pattern="add"),
            call("glob", pattern="*.py"),
            final("done"),
        ]
    )
    r = agent.run("x")
    assert r.status == "completed"
    assert tags(r) == ["overrun"]
    assert "2 turns after declaring completion at turn 0" in r.failures[0][1]


def test_verifying_once_after_claim_is_fine(make_agent):
    agent, _ = make_agent(
        [
            {"content": "All tests pass now.", **call("bash", cmd="true")},
            call("glob", pattern="x"),
            final("ok"),
        ]
    )
    assert agent.run("x").failures == []


# -- malformed -----------------------------------------------------------------------------------


def test_malformed_streak_aborts(make_agent):
    agent, _ = make_agent([call("nope")] * 3)
    r = agent.run("x")
    assert r.status == "failed"
    assert tags(r) == ["malformed_call"] * 4
    assert "3 consecutive turns" in r.failures[-1][1]


def test_empty_reply_is_malformed(make_agent):
    agent, server = make_agent(
        [
            {"reasoning": "thinking only"},
            {"expect": {"last_contains": "empty reply"}, **final("ok")},
        ]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["malformed_call"]


def test_salvaged_call_is_tagged_but_runs(make_agent):
    agent, _ = make_agent(
        [
            {
                "content": '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.md"}}\n</tool_call>'
            },
            final(),
        ]
    )
    r = agent.run("x")
    assert r.status == "completed"
    assert tags(r) == ["malformed_call"] and r.failures[0][1].startswith("salvaged")


@pytest.mark.parametrize("protocol", ["json_schema", "grammar"])
def test_loop_detection_under_constrained_protocols(make_agent, protocol):
    agent, server = make_agent(
        [call("glob", pattern="*.py")] * 3 + [final("ok")],
        [f"agent.protocol={json.dumps(protocol)}"],
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    assert tags(r) == ["loop"]


# -- units ---------------------------------------------------------------------------------------


def test_loop_detector_frequency():
    d = LoopDetector(repeat=3, abort=6)
    seq = ["a", "b", "a", "c", "a", "d", "a"]
    verdicts = [d.observe(x, {}, "r", x) for x in seq]
    assert verdicts[:6] == [None] * 6
    assert verdicts[6] and "4x in the last 7 calls" in verdicts[6].detail


def test_periodic_tail():
    assert periodic_tail(
        "intro " + "the same thing, " * 40, 400
    ) == "the same thing, " or periodic_tail("intro " + "the same thing, " * 40, 400)
    assert periodic_tail("x" * 1000, 400)
    assert periodic_tail("".join(f"line {i}\n" for i in range(200)), 400) is None
    assert periodic_tail("short", 400) is None


def test_role_marker_and_claims():
    assert cut_at_role_marker("Answer.\nUser: more?") == ("Answer.", "User:")
    assert cut_at_role_marker("<|im_start|>user") == ("<|im_start|>user", None)  # nothing before it
    assert cut_at_role_marker("use a user: field") == ("use a user: field", None)
    assert claims_completion("The task is now complete.")
    assert claims_completion("all tests pass")
    assert not claims_completion("Let me run the tests to see if they pass.")
    assert repeated_paragraph("short\n\nshort") is None


def test_failures_cli(make_agent, db_path, capsys):
    agent, _ = make_agent([call("glob", pattern="*.py")] * 3 + [final("ok")])
    agent.run("x")
    assert main(["failures", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "loop" in out and "1 runs" in out
