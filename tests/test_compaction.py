"""Context compaction with a separate small model (milestone 7)."""

from __future__ import annotations

import json

import pytest

from argus.compact import SUMMARY_PREFIX, is_tool_result, mask_message, turn_groups
from argus.mock import Script, call, final
from tests.conftest import q


@pytest.fixture
def big_files(workspace):
    for n in range(4):
        (workspace / f"big{n}.txt").write_text(
            "".join(f"line {i} of file {n} with several words in it\n" for i in range(150))
        )
    return workspace


def reads(n: int):
    return [call("read", path=f"big{i}.txt") for i in range(n)]


def check_pairing(messages):
    """Every native tool result must follow the assistant message that called it."""
    open_ids: set[str] = set()
    for m in messages:
        if m["role"] == "assistant":
            open_ids = {tc["id"] for tc in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in open_ids, m


def test_masking_old_tool_output(make_agent, big_files):
    agent, server = make_agent(
        reads(4) + [final("done")],
        ["compaction.keep_last_turns=1", "compaction.summarize=false"],
        server_kw={"n_ctx": 16000},
    )
    r = agent.run("read the big files")
    assert r.status == "completed", server.errors
    comps = agent.store.compactions(r.run_id)
    assert comps and comps[0]["stage"] == "mask"
    assert comps[0]["tokens_after"] < comps[0]["tokens_before"]
    last = server.requests[-1]["messages"]
    assert last[0]["role"] == "system" and last[1]["content"] == "read the big files"
    stubs = [
        m
        for m in last
        if m["role"] == "tool" and m["content"].startswith("[old tool output elided")
    ]
    assert stubs
    assert "line 0 of file 3" in last[-1]["content"]  # the most recent output is intact
    check_pairing(last)
    assert max(x["usage"]["prompt_tokens"] for x in server.responses) < 16000
    masked = [m for m in agent.store.messages(r.run_id) if m["kind"] == "masked"]
    assert masked
    # the turn logs show which (compacted) messages each request contained
    ids = json.loads(agent.store.turns(r.run_id)[-1]["context_ids"])
    assert set(m["id"] for m in masked) & set(ids)


def test_summarize_with_small_model(make_agent, mock, big_files):
    small = mock(Script.always({"content": "Read big0-big2; nothing changed yet."}))
    agent, server = make_agent(
        reads(4) + [final("done")],
        [
            "compaction.keep_last_turns=1",
            "compaction.mask=false",
            f"compaction.base_url={q(small.url)}",
            'compaction.model="tiny"',
        ],
        server_kw={"n_ctx": 16000},
    )
    r = agent.run("read the big files")
    assert r.status == "completed", server.errors
    comps = agent.store.compactions(r.run_id)
    assert comps[0]["stage"] == "summarize" and comps[0]["model"] == "tiny"
    assert comps[0]["summary"] == "Read big0-big2; nothing changed yet."
    req = small.requests[0]
    assert req["model"] == "tiny" and req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Task:\nread the big files" in req["messages"][1]["content"]
    assert "## result" in req["messages"][1]["content"]
    last = server.requests[-1]["messages"]
    assert last[2]["content"].startswith(SUMMARY_PREFIX)
    assert last[3]["role"] == "assistant"
    check_pairing(last)


def test_drop_when_small_model_unreachable(make_agent, big_files):
    agent, server = make_agent(
        reads(4) + [final("done")],
        [
            "compaction.keep_last_turns=1",
            "compaction.mask=false",
            'compaction.base_url="http://127.0.0.1:9/v1"',
        ],
        server_kw={"n_ctx": 16000},
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    comps = agent.store.compactions(r.run_id)
    assert comps[0]["stage"] == "drop"
    assert agent.store.events(r.run_id, "compaction_error")
    assert "earlier messages were removed" in server.requests[-1]["messages"][2]["content"]


def test_emergency_compaction_on_context_overflow(make_agent, big_files):
    agent, server = make_agent(
        reads(3) + [final("done")],
        ["compaction.keep_last_turns=1", "compaction.summarize=false"],
        server_kw={"n_ctx": 9000},
    )
    agent.compactor.limit = lambda n_ctx: (
        10**9
    )  # never compact proactively; the overflow error must
    r = agent.run("x")
    assert r.status == "completed", (r.failures, server.errors)
    assert ("token_cap" in {t for t, _ in r.failures}) and agent.store.compactions(r.run_id)


def test_compaction_disabled_overflow_fails(make_agent, big_files):
    agent, _ = make_agent(
        reads(3) + [final()], ["compaction.enabled=false"], server_kw={"n_ctx": 9000}
    )
    r = agent.run("x")
    assert r.status == "failed" and r.failures[-1][0] == "token_cap"


def test_masking_under_constrained_protocol(make_agent, big_files):
    agent, server = make_agent(
        reads(4) + [final("done")],
        [
            'agent.protocol="json_schema"',
            "compaction.keep_last_turns=1",
            "compaction.summarize=false",
        ],
        server_kw={"n_ctx": 12000},
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    last = server.requests[-1]["messages"]
    stubs = [m for m in last if m["role"] == "user" and "[old tool output elided" in m["content"]]
    assert stubs and all(m["content"].startswith("<tool_response>\n") for m in stubs)


# -- units ---------------------------------------------------------------------------------------


def test_turn_groups():
    ctx = [(i, m) for i, m in enumerate([
        {"role": "system"}, {"role": "user"},
        {"role": "assistant"}, {"role": "tool"}, {"role": "tool"},
        {"role": "assistant"}, {"role": "user"},
        {"role": "assistant"},
    ])]  # fmt: skip
    assert turn_groups(ctx) == [(2, 5), (5, 7), (7, 8)]


def test_mask_message_variants():
    tool = {"role": "tool", "tool_call_id": "c1", "content": "x\n" * 300}
    masked = mask_message(tool, 100)
    assert masked["tool_call_id"] == "c1" and masked["content"].startswith(
        "[old tool output elided"
    )
    assert mask_message({"role": "tool", "content": "short"}, 100) is None
    user_tool = {"role": "user", "content": "<tool_response>\n" + "y" * 500 + "\n</tool_response>"}
    assert is_tool_result(user_tool)
    assert mask_message(user_tool, 100)["content"].endswith("</tool_response>")
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps({"path": "a", "new": "z" * 1000}),
                },
            }
        ],
    }
    args = json.loads(mask_message(asst, 100)["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "a" and "chars elided" in args["new"]
    envelope = {
        "role": "assistant",
        "content": json.dumps({"tool": "edit", "args": {"new": "q" * 800}}),
    }
    assert "chars elided" in json.loads(mask_message(envelope, 100)["content"])["args"]["new"]
