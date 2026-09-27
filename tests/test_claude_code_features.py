"""Todo list, plan mode, subagents, hooks and progressive-disclosure skills."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from argus.console import render_run
from argus.mock import call, final
from argus.sandbox import choose
from argus.subagents import EXPLORE_PROMPT

from .test_context import write_skill

TODO = [
    {"content": "read calc.py", "status": "done"},
    {"content": "fix add", "status": "in_progress"},
    {"content": "run tests", "status": "pending"},
]


def tool_names(req):
    tools = req.get("tools") or []
    return sorted(t["function"]["name"] for t in tools)


# -- todo -------------------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["native", "grammar"])
def test_todo_tool(make_agent, protocol):
    agent, server = make_agent(
        [call("todo", items=TODO), final("ok")],
        overrides=["tools.todo='on'", f"agent.protocol='{protocol}'"],
    )
    result = agent.run("fix add")
    assert result.status == "completed" and not server.errors
    tc = agent.store.tool_calls(result.run_id)[0]
    assert "[x] read calc.py\n[>] fix add\n[ ] run tests" in tc["result"]
    assert "(1/3 done)" in tc["result"]
    ev = agent.store.events(result.run_id, "todo")
    assert json.loads(ev[0]["data_json"]) == TODO
    assert "[>] fix add" in render_run(agent.store, result.run_id, color=False)


def test_todo_is_off_for_local_models_and_on_for_frontier(make_agent, monkeypatch):
    agent, server = make_agent([final()])
    agent.run("x")
    assert "todo" not in tool_names(server.requests[0])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-000000000")
    agent, server = make_agent(
        [final()], overrides=['model.provider="anthropic"', 'model.model="claude-sonnet-5"']
    )
    agent.run("x")
    names = [t["name"] for t in server.requests[0]["tools"]]
    assert "todo" in names and "task" in names


# -- plan mode ----------------------------------------------------------------------------------


@pytest.mark.skipif(choose("auto").name == "none", reason="no sandbox here")
def test_plan_mode(make_agent, workspace):
    steps = [
        call("bash", cmd="cat calc.py; echo hacked > calc.py"),
        final("1. In calc.py change `a - b` to `a + b`.\n2. Run the tests."),
    ]
    agent, server = make_agent(steps, overrides=['agent.mode="plan"'])
    result = agent.run("fix add")
    assert result.status == "completed" and result.final.startswith("1. In calc.py")
    req = server.requests[0]
    assert "edit" not in tool_names(req) and "bash" in tool_names(req)
    assert "You are in plan mode" in req["messages"][0]["content"]
    assert "return a - b" in (workspace / "calc.py").read_text()  # nothing changed
    assert agent.store.approvals(result.run_id)[0]["sandbox"] == "read-only"
    assert agent.store.run(result.run_id)["mode"] == "plan"


def test_plan_mode_without_sandbox_drops_bash(make_agent):
    agent, server = make_agent(
        [final("plan")], overrides=['agent.mode="plan"', 'sandbox.backend="none"']
    )
    agent.run("x")
    assert tool_names(server.requests[0]) == ["glob", "grep", "read"]


# -- subagents -----------------------------------------------------------------------------------


def write_agent(workspace: Path, name: str, front: str, body: str) -> None:
    d = workspace / ".argus" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(f"---\n{front}\n---\n{body}\n")


def test_explore_subagent(make_agent, workspace):
    steps = [
        call("task", agent="explore", prompt="Where is add defined?"),
        call("grep", pattern="def add"),  # the subagent's turns
        final("add is defined in calc.py:1"),
        final("It is in calc.py."),  # the parent again
    ]
    agent, server = make_agent(steps, overrides=['agent.subagents="on"'])
    result = agent.run("where is add?")
    assert result.status == "completed", result.failures
    parent_req, child_req = server.requests[0], server.requests[1]
    assert "task" in tool_names(parent_req)
    assert tool_names(child_req) == ["bash", "glob", "grep", "read"]
    assert EXPLORE_PROMPT[:40] in child_req["messages"][0]["content"]
    assert child_req["messages"][1]["content"] == "Where is add defined?"  # a fresh context
    tc = agent.store.tool_calls(result.run_id)[0]
    assert tc["result"].startswith("add is defined in calc.py:1\n[subagent explore: completed")
    (child,) = agent.store.children(result.run_id)
    assert (child["agent"], child["parent_turn"], child["status"]) == ("explore", 0, "completed")
    assert server.requests[3]["messages"][-1]["content"].startswith("add is defined")


def test_custom_subagent_with_its_own_model(make_agent, workspace):
    write_agent(
        workspace,
        "reviewer",
        "name: reviewer\ndescription: reviews diffs\ntools: read, grep\nmodel: llama/tiny-reviewer\nmax_turns: 5",
        "You review code.",
    )
    steps = [
        call("task", agent="reviewer", prompt="review calc.py"),
        final("looks fine"),
        final("done"),
    ]
    agent, server = make_agent(steps, overrides=['agent.subagents="on"'])
    result = agent.run("review")
    assert result.status == "completed"
    enum = server.requests[0]["tools"]
    task = next(t for t in enum if t["function"]["name"] == "task")
    assert task["function"]["parameters"]["properties"]["agent"]["enum"] == [
        "explore",
        "general",
        "reviewer",
    ]
    child_req = server.requests[1]
    assert child_req["model"] == "tiny-reviewer"
    assert tool_names(child_req) == ["grep", "read"]
    assert "You review code." in child_req["messages"][0]["content"]


def test_subagent_cannot_be_looser_than_its_parent(make_agent, workspace):
    steps = [
        call("task", agent="general", prompt="fix add in calc.py"),
        call("edit", path="calc.py", old="return a - b", new="return a + b"),
        final("could not edit"),
        final("the subagent could not edit"),
    ]
    agent, _ = make_agent(
        steps,
        overrides=[
            'agent.subagents="on"',
            'agent.approval="read-only"',
            "tools.require_read_before_edit=false",
        ],
    )
    result = agent.run("fix")
    (child,) = agent.store.children(result.run_id)
    assert "not allowed" in agent.store.tool_calls(child["id"])[0]["result"]
    assert "return a - b" in (workspace / "calc.py").read_text()


# -- hooks -----------------------------------------------------------------------------------------


def hook_script(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(
        "#!/usr/bin/env python3\nimport json, sys, os\nev = json.load(sys.stdin)\n"
        + textwrap.dedent(body)
    )
    p.chmod(0o755)
    return str(p)


def hooks_config(tmp_path: Path, *hooks: dict) -> Path:
    lines = []
    for h in hooks:
        lines.append("[[hooks]]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in h.items()]
    path = tmp_path / "hooks.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def agent_with_config(make_agent, steps, cfg_path: Path, *overrides):
    from argus.agent import Agent
    from argus.config import load_config

    base, server = make_agent(steps, overrides=list(overrides))
    cfg = load_config(cfg_path, [f"{k}={json.dumps(v)}" for k, v in [
        ("model.base_url", server.url), ("executor.workdir", base.cfg.executor.workdir),
        ("log.db", str(base.store.path)), ("model.retries", 0),
    ]] + list(overrides))  # fmt: skip
    return Agent(cfg, store=base.store), server


def test_pre_and_post_tool_hooks(make_agent, tmp_path):
    guard = hook_script(
        tmp_path,
        "guard.py",
        """
        if "rm " in ev["args"].get("cmd", ""):
            print("destructive commands are not allowed here", file=sys.stderr)
            sys.exit(2)
        """,
    )
    note = hook_script(tmp_path, "note.py", 'print("formatted " + ev["tool"])\n')
    cfg = hooks_config(
        tmp_path,
        {"event": "pre_tool", "matcher": "bash", "command": guard},
        {"event": "post_tool", "matcher": "edit|bash", "command": note},
    )
    steps = [call("bash", cmd="rm -rf pkg"), call("bash", cmd="echo hi"), final("ok")]
    agent, _ = agent_with_config(make_agent, steps, cfg)
    try:
        result = agent.run("x")
    finally:
        agent.close()
    first, second = agent.store.tool_calls(result.run_id)
    assert "blocked by a hook: destructive commands are not allowed here" in first["result"]
    assert "hi" in second["result"] and "[hook: formatted bash]" in second["result"]
    events = [json.loads(e["data_json"]) for e in agent.store.events(result.run_id, "hook")]
    assert [(e["event"], e["code"]) for e in events] == [
        ("pre_tool", 2),
        ("pre_tool", 0),
        ("post_tool", 0),
    ]


def test_pre_tool_hook_decision_answers_the_approval(make_agent, tmp_path, workspace):
    allow = hook_script(tmp_path, "allow.py", 'print(json.dumps({"decision": "allow"}))\n')
    cfg = hooks_config(tmp_path, {"event": "pre_tool", "matcher": "edit", "command": allow})
    steps = [call("edit", path="calc.py", old="return a - b", new="return a + b"), final("ok")]
    agent, _ = agent_with_config(
        make_agent, steps, cfg, 'agent.approval="ask"', "tools.require_read_before_edit=false"
    )
    try:
        result = agent.run("fix")  # the headless approver would say no
    finally:
        agent.close()
    assert "return a + b" in (workspace / "calc.py").read_text()
    row = agent.store.approvals(result.run_id)[0]
    assert (row["allowed"], row["source"]) == (1, "hook")


def test_stop_hook_sends_the_model_back(make_agent, tmp_path):
    marker = tmp_path / "blocked-once"
    stop = hook_script(
        tmp_path,
        "stop.py",
        f"""
        if not os.path.exists({str(marker)!r}):
            open({str(marker)!r}, "w").close()
            print("run the tests before finishing", file=sys.stderr)
            sys.exit(2)
        """,
    )
    cfg = hooks_config(tmp_path, {"event": "stop", "command": stop})
    steps = [final("done?"), call("bash", cmd="echo tests pass"), final("done, tests pass")]
    agent, server = agent_with_config(make_agent, steps, cfg)
    try:
        result = agent.run("x")
    finally:
        agent.close()
    assert result.status == "completed" and result.final == "done, tests pass"
    assert server.requests[1]["messages"][-1]["content"] == "run the tests before finishing"


def test_session_and_prompt_hooks(make_agent, tmp_path):
    ctx = hook_script(tmp_path, "ctx.py", 'print("Current branch: main")\n')
    prompt = hook_script(
        tmp_path,
        "prompt.py",
        """
        if "secret" in ev["prompt"]:
            print("prompt mentions a secret", file=sys.stderr)
            sys.exit(2)
        print("ticket: ARG-42")
        """,
    )
    cfg = hooks_config(
        tmp_path,
        {"event": "session_start", "command": ctx},
        {"event": "user_prompt", "command": prompt},
    )
    agent, server = agent_with_config(make_agent, [final("ok")], cfg)
    try:
        agent.run("fix it")
        blocked = agent.run("print the secret")
    finally:
        agent.close()
    msgs = server.requests[0]["messages"]
    assert "Current branch: main" in msgs[0]["content"]
    assert msgs[1]["content"] == "fix it\n\nticket: ARG-42"
    assert blocked.status == "failed" and blocked.failures[0][0] == "blocked"
    assert len(server.requests) == 1  # the blocked run never reached the model


# -- progressive skills ---------------------------------------------------------------------------


def test_skill_files_level_three(make_agent, workspace):
    write_skill(
        workspace / ".agents/skills",
        "release",
        "Cut a release.",
        "# Release\nSee reference/steps.md.",
        {"reference/steps.md": "1. bump\n2. tag\n"},
    )
    steps = [
        call("skill", name="release"),
        call("skill", name="release", file="reference/steps.md"),
        call("skill", name="release", file="../../calc.py"),
        final("ok"),
    ]
    agent, _ = make_agent(steps)
    result = agent.run("release")
    level2, level3, escape = agent.store.tool_calls(result.run_id)
    assert "files: reference/steps.md" in level2["result"]
    assert level3["result"] == "1. bump\n2. tag\n"
    assert "outside the skill directory" in escape["result"]
    vias = [r["via"] for r in agent.store.skill_invocations(result.run_id)]
    assert vias == ["tool", "file"]
    assert os.path.exists(workspace / ".agents/skills/release/reference/steps.md")
