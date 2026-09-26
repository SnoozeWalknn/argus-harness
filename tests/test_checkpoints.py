"""Shadow-git checkpoints and post-tool filesystem checks (milestone 5)."""

from __future__ import annotations

import json
import subprocess

from argus.cli import main
from argus.config import CheckpointConfig
from argus.executors import LocalExecutor
from argus.fsstate import Checkpointer, ShadowGit
from argus.mock import call, final
from tests.conftest import CALC, git_init
from tests.test_agent import fix_script


def git(path, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    ).stdout


def test_checkpoint_chain_and_changes(make_agent, workspace):
    agent, server = make_agent(fix_script())
    r = agent.run("fix")
    assert r.status == "completed", server.errors
    cps = agent.store.checkpoints(r.run_id)
    assert [c["reason"] for c in cps] == ["baseline", "before edit", "before bash", "final"]
    # nothing changed between baseline and the edit, so the same commit is reused
    assert cps[0]["commit_sha"] == cps[1]["commit_sha"]
    assert cps[1]["tree_sha"] != cps[2]["tree_sha"] == cps[3]["tree_sha"]
    calls = agent.store.tool_calls(r.run_id)
    edit = next(c for c in calls if c["name"] == "edit")
    assert json.loads(edit["fs_changes_json"]) == [["M", "calc.py"]]
    assert edit["checkpoint"] == cps[1]["commit_sha"]
    read = next(c for c in calls if c["name"] == "read")
    assert read["checkpoint"] is None  # read-only tools are never snapshotted
    assert agent.store.failures(r.run_id) == []


def test_bash_changes_reported_and_mark_files_stale(make_agent, workspace):
    agent, server = make_agent(
        [
            call("read", path="calc.py"),
            call("bash", cmd="sed -i 's/a - b/a + b/' calc.py && echo hi > notes.txt"),
            {
                "expect": {"last_contains": "[files changed: M calc.py, A notes.txt]"},
                **call("edit", path="calc.py", old="return a - b\n    # x", new="y"),
            },
            {"expect": {"last_contains": "changed since you last read it"}, **final()},
        ]
    )
    r = agent.run("x")
    assert r.status == "completed", server.errors
    bash = agent.store.tool_calls(r.run_id)[1]
    assert json.loads(bash["fs_changes_json"]) == [["M", "calc.py"], ["A", "notes.txt"]]


def test_mass_deletion_flagged(make_agent, workspace):
    for i in range(25):
        (workspace / f"f{i}.txt").write_text(str(i))
    agent, _ = make_agent([call("bash", cmd="rm -f f*.txt"), final()])
    r = agent.run("x")
    assert ("fs_violation", r.failures[0][1]) == r.failures[0]
    assert "deleted 25 files" in r.failures[0][1]


def test_edit_touching_other_files_is_a_violation(make_agent, workspace):
    agent, _ = make_agent(
        [
            call("read", path="calc.py"),
            call("edit", path="calc.py", old="a - b", new="a + b"),
            final(),
        ]
    )
    edit = agent.tools["edit"]
    original = edit.run

    def rogue(ctx, args):
        (workspace / "rogue.txt").write_text("surprise")
        return original(ctx, args)

    edit.run = rogue
    r = agent.run("x")
    assert r.failures == [("fs_violation", "edit of calc.py also changed: A rogue.txt")]


def test_edit_in_ignored_dir_is_not_a_violation(make_agent, workspace):
    (workspace / ".gitignore").write_text("build/\n")
    agent, _ = make_agent([call("edit", path="build/out.txt", old="", new="x\n"), final()])
    r = agent.run("x")
    assert r.status == "completed" and r.failures == []
    assert json.loads(agent.store.tool_calls(r.run_id)[0]["meta_json"])["untracked"] is True


def test_user_repository_untouched(make_agent, workspace):
    git_init(workspace)
    head = git(workspace, "rev-parse", "HEAD")
    refs = git(workspace, "for-each-ref")
    agent, _ = make_agent(fix_script())
    assert agent.run("fix").status == "completed"
    assert git(workspace, "rev-parse", "HEAD") == head
    assert git(workspace, "for-each-ref") == refs
    assert git(workspace, "diff", "--cached") == ""  # user's index untouched
    assert git(workspace, "status", "--short") == " M calc.py\n"
    assert git(workspace, "stash", "list") == ""


def test_no_git_disables_checkpoints(make_agent, workspace):
    agent, _ = make_agent(fix_script())
    agent.executor._commands["git"] = False
    r = agent.run("fix")
    assert r.status == "completed"
    assert agent.store.checkpoints(r.run_id) == []
    events = agent.store.events(r.run_id, "checkpoints_disabled")
    assert "git is not installed" in events[0]["data_json"]


def test_checkpoints_can_be_turned_off(make_agent):
    agent, _ = make_agent(fix_script(), ["checkpoint.enabled=false"])
    r = agent.run("fix")
    assert r.status == "completed" and agent.store.checkpoints(r.run_id) == []


def test_cli_diff_restore_checkpoints(make_agent, workspace, db_path, capsys):
    agent, _ = make_agent(
        fix_script()[:2]
        + [call("bash", cmd="mkdir -p out && echo data > out/new.txt && rm README.md"), final()]
    )
    r = agent.run("fix")
    db = ["--db", str(db_path)]
    assert main(["checkpoints", r.run_id, *db]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "M calc.py" in out

    assert main(["diff", r.run_id, "--stat", *db]) == 0
    assert capsys.readouterr().out.split("\n")[:3] == ["D README.md", "M calc.py", "A out/new.txt"]
    assert main(["diff", r.run_id, *db]) == 0
    assert "+    return a + b" in capsys.readouterr().out

    (workspace / "later.txt").write_text("user work after the run")
    assert main(["restore", r.run_id, *db]) == 1  # dry run
    assert "delete later.txt" in capsys.readouterr().out
    assert (workspace / "later.txt").exists()

    assert main(["restore", r.run_id, "--yes", *db]) == 0
    assert "previous state saved" in capsys.readouterr().out
    assert (workspace / "calc.py").read_text() == CALC
    assert (workspace / "README.md").exists()
    assert not (workspace / "out/new.txt").exists() and not (workspace / "later.txt").exists()

    # the restore itself can be undone from the saved state
    assert main(["restore", r.run_id, "--to", "final", "--yes", *db]) == 0
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_shadow_repo_is_outside_workspace(workspace, tmp_path):
    ex = LocalExecutor(str(workspace))
    ck = Checkpointer(ex, CheckpointConfig(), "r1")
    cp = ck.start()
    assert cp and ck.active
    assert not (workspace / ".git").exists()
    assert ck.git.git_dir.startswith(str(tmp_path / "shadow"))
    assert ShadowGit(ex).git_dir == ck.git.git_dir  # stable per workspace
