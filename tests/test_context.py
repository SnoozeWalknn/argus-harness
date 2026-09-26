"""AGENTS.md and skills (milestone 6)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from argus.mock import call, final
from argus.skills import parse_frontmatter, render_skill_index


def write_skill(
    root: Path, name: str, description: str, body: str, extra: dict | None = None
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    for fname, content in (extra or {}).items():
        (d / fname).parent.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(content)
    return d / "SKILL.md"


@pytest.fixture
def skilled(workspace):
    write_skill(
        workspace / ".agents/skills",
        "release",
        "Cut a release: bump the version, update CHANGELOG.md, tag.",
        "# Release\n1. Bump version in pyproject.toml\n2. Run scripts/tag.sh",
        {"scripts/tag.sh": "git tag v$1\n"},
    )
    home = Path(os.environ["HOME"])
    write_skill(
        home / ".config/argus/skills",
        "house-style",
        "Write commit messages in house style.",
        "Use imperative mood.",
    )
    (workspace / "AGENTS.md").write_text(
        "Run tests with `python3 -m pytest -q`.\nNever edit vendored code.\n"
    )
    (home / ".config/argus/AGENTS.md").write_text("Prefer small diffs.\n")
    return workspace


def test_agents_md_and_skill_index_in_prompt(make_agent, skilled):
    agent, server = make_agent([final()])
    r = agent.run("x")
    system = server.requests[0]["messages"][0]["content"]
    assert "Prefer small diffs." in system
    assert "Instructions from AGENTS.md:\nRun tests with" in system
    assert "- release: Cut a release: bump the version" in system
    assert "- house-style: Write commit messages" in system
    assert "Use imperative mood" not in system  # bodies load on demand
    tools = {t["function"]["name"]: t for t in server.requests[0]["tools"]}
    assert tools["skill"]["function"]["parameters"]["properties"]["name"]["enum"] == [
        "release",
        "house-style",
    ]
    ov = json.loads(agent.store.run(r.run_id)["overhead_json"])
    assert ov["parts"]["agents_md"] > 0 and ov["parts"]["skills"] > 0
    ctx = json.loads(agent.store.events(r.run_id, "context")[0]["data_json"])
    assert ctx["agents_md"][-1] == "AGENTS.md" and len(ctx["skills"]) == 2


def test_skill_tool_invocation_logged(make_agent, skilled):
    agent, server = make_agent(
        [
            call("skill", name="release"),
            {
                "expect": {"last_contains": "1. Bump version in pyproject.toml"},
                **call("skill", name="house-style"),
            },
            {"expect": {"last_contains": "on the argus host"}, **final()},
        ]
    )
    r = agent.run("cut a release")
    assert r.status == "completed", server.errors
    result = agent.store.tool_calls(r.run_id)[0]["result"]
    assert "files: scripts/tag.sh" in result
    inv = agent.store.skill_invocations(r.run_id)
    assert [(i["skill"], i["via"], i["turn_idx"]) for i in inv] == [
        ("release", "tool", 0),
        ("house-style", "tool", 1),
    ]


def test_skill_read_directly_is_logged(make_agent, skilled):
    agent, _ = make_agent(
        [
            call("read", path=".agents/skills/release/SKILL.md"),
            call("bash", cmd="cat .agents/skills/release/SKILL.md"),
            final(),
        ]
    )
    r = agent.run("x")
    inv = agent.store.skill_invocations(r.run_id)
    assert [(i["skill"], i["via"]) for i in inv] == [("release", "read"), ("release", "bash")]


def test_unknown_skill_rejected_natively(make_agent, skilled):
    agent, server = make_agent([call("skill", name="nope"), final()])
    r = agent.run("x")
    assert (
        r.failures[0][0] == "malformed_call"
        and "must be one of 'release', 'house-style'" in r.failures[0][1]
    )


@pytest.mark.parametrize("protocol", ["json_schema", "grammar"])
def test_skill_under_constrained_protocols(make_agent, skilled, protocol):
    agent, server = make_agent(
        [call("skill", name="release"), final()], [f"agent.protocol={json.dumps(protocol)}"]
    )
    r = agent.run("x")
    assert r.status == "completed" and not server.errors
    assert (
        "skill(name) - load a skill's instructions" in server.requests[0]["messages"][0]["content"]
    )
    # a skill name outside the enum cannot be produced under the constraint
    agent2, server2 = make_agent(
        [call("skill", name="nope"), final()], [f"agent.protocol={json.dumps(protocol)}"]
    )
    agent2.run("x")
    assert any("violates the request constraint" in e for e in server2.errors)


def test_no_skills_no_tool(make_agent):
    agent, server = make_agent([final()])
    agent.run("x")
    assert "skill" not in [t["function"]["name"] for t in server.requests[0]["tools"]]
    assert "Skills" not in server.requests[0]["messages"][0]["content"]


def test_context_can_be_disabled(make_agent, skilled):
    agent, server = make_agent([final()], ["context.agents_md=false", "context.skills=false"])
    agent.run("x")
    system = server.requests[0]["messages"][0]["content"]
    assert "AGENTS.md" not in system and "Skills" not in system


def test_agents_md_truncated(make_agent, workspace):
    (workspace / "AGENTS.md").write_text("x" * 5000)
    agent, server = make_agent([final()], ["context.max_agents_md_chars=100"])
    agent.run("x")
    assert (
        "[... truncated; read AGENTS.md for the rest]"
        in server.requests[0]["messages"][0]["content"]
    )


def test_first_skill_dir_wins(make_agent, workspace):
    write_skill(workspace / ".agents/skills", "dup", "from agents", "A")
    write_skill(workspace / ".claude/skills", "dup", "from claude", "B")
    agent, _ = make_agent([final()])
    assert [(s.name, s.description) for s in agent.skills] == [("dup", "from agents")]


def test_parse_frontmatter():
    meta, body = parse_frontmatter(
        "---\nname: pdf\ndescription: >\n  Extract text\n  from PDFs.\nlicense: 'MIT'\nallowed-tools:\n  - bash\n---\n# Body\n"
    )
    assert meta == {
        "name": "pdf",
        "description": "Extract text from PDFs.",
        "license": "MIT",
        "allowed-tools": "",
    }
    assert body == "# Body\n"
    assert parse_frontmatter("# no front matter") == ({}, "# no front matter")
    assert parse_frontmatter("---\nunterminated: yes\n") == ({}, "---\nunterminated: yes\n")


def test_skill_index_rendering():
    assert render_skill_index([]) == ""
