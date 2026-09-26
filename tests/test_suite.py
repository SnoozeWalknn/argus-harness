"""Task suites, the completion oracle, A/B runs and reports (milestone 8).

Variant order is shuffled per task, so the mock models here are *policies*:
their reply depends on the conversation, not on the order of requests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from argus.cli import main
from argus.compact import is_tool_result
from argus.config import load_config
from argus.mock import Script, call, final
from argus.report import format_report, mcnemar_exact, summarize, wilson
from argus.store import Store
from argus.suite import SuiteError, SuiteRunner, append_task, load_suite
from tests.conftest import CALC, TEST_CALC, q

CHECK = "python3 -c 'import calc; assert calc.add(2, 3) == 5'"


def good_policy(extra_turns: int = 0):
    def step(req, server):
        msgs = req["messages"]
        task = msgs[1]["content"]
        if "Describe" in task:
            return final("A tiny calculator.")
        results = [m["content"] for m in msgs if is_tool_result(m)]
        if not results:
            return call("read", path="calc.py")
        if len(results) == 1:
            return call("edit", path="calc.py", old="return a - b", new="return a + b")
        if len(results) == 2:
            return call("bash", cmd=CHECK + " && echo ok")
        if len(results) < 3 + extra_turns:  # dawdling after the task is already done
            return call("grep", pattern=f"def|{len(results)}")
        return final("Fixed add().")

    return Script.always(step)


def bad_policy():
    return Script.always(lambda req, server: call("glob", pattern="*.py"))


@pytest.fixture
def suite_file(tmp_path) -> Path:
    fx = tmp_path / "fixtures" / "calc"
    fx.mkdir(parents=True)
    (fx / "calc.py").write_text(CALC)
    (fx / "test_calc.py").write_text(TEST_CALC)
    (fx / "README.md").write_text("# calc\n")
    p = tmp_path / "suite.toml"
    p.write_text(
        f"""
name = "smoke"

[defaults]
workspace = "fixtures/calc"
check_timeout = 60

[[task]]
id = "fix-add"
prompt = "Fix add() in calc.py."
check = {json.dumps(CHECK)}

[[task]]
id = "describe"
prompt = "Describe the project."
check = "test -f README.md"
setup = ["echo prepared > setup.txt"]
"""
    )
    return p


def cfg_for(server, db, *extra):
    return load_config(
        None, [f"model.base_url={q(server.url)}", f"log.db={q(str(db))}", "model.retries=0", *extra]
    )


def test_load_suite(suite_file):
    s = load_suite(suite_file)
    assert s.name == "smoke" and [t.id for t in s.tasks] == ["fix-add", "describe"]
    assert s.tasks[0].workspace.endswith("fixtures/calc") and s.tasks[0].check_timeout == 60
    assert s.tasks[1].setup == ["echo prepared > setup.txt"]
    assert s.select(["describe"])[0].id == "describe"
    with pytest.raises(SuiteError, match="unknown task"):
        s.select(["nope"])


@pytest.mark.parametrize(
    "body,err",
    [
        ('[[task]]\nprompt = "x"\nbogus = 1\n', "unknown key"),
        ('[[task]]\nid = "a"\n', "no prompt"),
        ('[[task]]\nid = "a"\nprompt = "x"\n[[task]]\nid = "a"\nprompt = "y"\n', "duplicate"),
        ('[[task]]\nprompt = "x"\nworkspace = "missing"\n', "not a directory"),
        ('name = "empty"\n', "no \\[\\[task\\]\\]"),
        (
            '[[task]]\nprompt = "x"\noverrides = ["agent.bogus=1"]\n',
            "unknown config key: agent.bogus",
        ),
    ],
)
def test_suite_validation(tmp_path, body, err):
    p = tmp_path / "s.toml"
    p.write_text(body)
    with pytest.raises(SuiteError, match=err):
        load_suite(p)


def test_suite_run_fresh_workspaces_and_checks(mock, suite_file, tmp_path):
    server = mock(good_policy())
    db = tmp_path / "db.sqlite"
    store = Store(db)
    runner = SuiteRunner(
        load_suite(suite_file), [("base", cfg_for(server, db))], store, repeat=2, keep=True
    )
    batch, outcomes = runner.run()
    assert [(o.task, o.passed) for o in outcomes] == [("fix-add", True), ("describe", True)] * 2
    # every run got its own copy of the fixture; the fixture itself is untouched
    assert len({o.workdir for o in outcomes}) == 4
    assert "return a - b" in (suite_file.parent / "fixtures/calc/calc.py").read_text()
    fixed = next(o for o in outcomes if o.task == "fix-add")
    assert "return a + b" in Path(fixed.workdir, "calc.py").read_text()
    described = next(o for o in outcomes if o.task == "describe")
    assert Path(described.workdir, "setup.txt").read_text() == "prepared\n"
    rows = store.runs(batch_id=batch)
    assert {(r["task_id"], r["rep"], r["variant"], r["check_passed"]) for r in rows} == {
        ("fix-add", 0, "base", 1),
        ("describe", 0, "base", 1),
        ("fix-add", 1, "base", 1),
        ("describe", 1, "base", 1),
    }


def test_passing_workspaces_removed_unless_keep(mock, suite_file, tmp_path):
    server = mock(good_policy())
    db = tmp_path / "db.sqlite"
    _, outcomes = SuiteRunner(
        load_suite(suite_file), [("base", cfg_for(server, db))], Store(db)
    ).run()
    assert all(o.passed and not Path(o.workdir).exists() for o in outcomes)


def test_failing_run_check_and_kept_workspace(mock, suite_file, tmp_path):
    server = mock(bad_policy())
    db = tmp_path / "db.sqlite"
    store = Store(db)
    batch, outcomes = SuiteRunner(
        load_suite(suite_file), [("bad", cfg_for(server, db))], store, task_ids=["fix-add"]
    ).run()
    (o,) = outcomes
    assert not o.passed and Path(o.workdir).exists()
    row = store.runs(batch_id=batch)[0]
    assert row["check_passed"] == 0 and "AssertionError" in row["check_output"]
    assert row["status"] == "failed"


def test_oracle_tags_overrun(mock, suite_file, tmp_path):
    server = mock(good_policy(extra_turns=3))
    db = tmp_path / "db.sqlite"
    store = Store(db)
    batch, outcomes = SuiteRunner(
        load_suite(suite_file),
        [("base", cfg_for(server, db))],
        store,
        task_ids=["fix-add"],
        oracle=True,
    ).run()
    assert outcomes[0].passed
    run_id = store.runs(batch_id=batch)[0]["id"]
    fails = store.failures(run_id)
    assert [f["tag"] for f in fails] == ["overrun"]
    # passed after the edit (turn 1); bash + 3 greps followed before the final answer
    assert "passed after turn 1; the agent continued 4 more tool turns" in fails[0]["detail"]
    oracle = [json.loads(e["data_json"]) for e in store.events(run_id, "oracle")]
    assert oracle[0] == {"passed": True}  # checked after the edit turn


def test_setup_failure_is_reported_not_raised(mock, tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[[task]]\nid = "broken"\nprompt = "x"\nsetup = ["exit 3"]\n')
    server = mock(good_policy())
    db = tmp_path / "db.sqlite"
    _, outcomes = SuiteRunner(load_suite(p), [("base", cfg_for(server, db))], Store(db)).run()
    assert outcomes[0].error.startswith("setup: setup command failed (3)")


def test_suite_over_ssh_shim(mock, suite_file, tmp_path):
    shim = [sys.executable, str(Path(__file__).parent / "fake_ssh.py")]
    server = mock(good_policy())
    db = tmp_path / "db.sqlite"
    cfg = cfg_for(server, db, 'executor.kind="ssh"', 'executor.host="vm"', 'executor.workdir="/tmp"',
                  f"executor.ssh_command={json.dumps(shim)}")  # fmt: skip
    store = Store(db)
    batch, outcomes = SuiteRunner(
        load_suite(suite_file), [("vm", cfg)], store, task_ids=["fix-add"], keep=True
    ).run()
    assert outcomes[0].passed
    assert store.runs(batch_id=batch)[0]["executor"].startswith("ssh:vm:/tmp/argus-fix-add-")


def test_ab_report_and_mcnemar(mock, suite_file, tmp_path, capsys):
    good, bad = mock(good_policy()), mock(bad_policy())
    db = tmp_path / "db.sqlite"
    a = tmp_path / "good.toml"
    b = tmp_path / "bad.toml"
    a.write_text(f'[model]\nbase_url = "{good.url}"\nretries = 0\n')
    b.write_text(f'[model]\nbase_url = "{bad.url}"\nretries = 0\n')
    rc = main(["ab", str(a), str(b), "--suite", str(suite_file), "-n", "2", "--db", str(db), "-q",
               "-t", "fix-add", "--ob", "agent.protocol=\"grammar\""])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 1  # some runs failed
    assert "good" in out and "bad" in out
    assert "only good passed: 2, only bad passed: 0" in out
    assert "exact McNemar p = 0.500" in out
    store = Store(db)
    batch = store.resolve_batch("last")
    s = summarize(store, batch)
    by = {v["label"]: v for v in s["variants"]}
    assert by["good"]["passes"] == 2 and by["bad"]["passes"] == 0
    assert by["bad"]["failure_runs"] == {"loop": 2}
    rows = store.runs(batch_id=batch)
    assert {r["protocol"] for r in rows if r["variant"] == "bad"} == {"grammar"}
    assert main(["report", "--db", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["paired"]["only_a_passed"] == 2


def test_ab_same_config_gets_labels(mock, suite_file, tmp_path, capsys):
    server = mock(good_policy())
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[model]\nbase_url = "{server.url}"\n')
    rc = main(["ab", str(cfg), str(cfg), "--suite", str(suite_file), "--db", str(tmp_path / "d.db"), "-q",
               "--oa", "agent.protocol=\"native\"", "--ob", "agent.protocol=\"json_schema\""])  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    assert "c-A" in out and "c-B" in out


def test_suite_cli_and_add(mock, suite_file, tmp_path, capsys, workspace):
    server = mock(good_policy())
    db = tmp_path / "db.sqlite"
    rc = main(
        [
            "suite",
            "run",
            str(suite_file),
            "-o",
            f"model.base_url={q(server.url)}",
            "--db",
            str(db),
            "-q",
        ]
    )
    assert rc == 0
    assert "2/2" in capsys.readouterr().out
    assert main(["suite", "list", str(suite_file)]) == 0
    assert "fix-add" in capsys.readouterr().out

    new = tmp_path / "saved.toml"
    assert main(["suite", "add", str(new), "--run", "last", "--db", str(db), "--id", "again",
                 "--check", "true", "--workspace", str(workspace)]) == 0  # fmt: skip
    saved = load_suite(new)
    assert saved.tasks[0].id == "again" and saved.tasks[0].check == "true"
    assert saved.tasks[0].prompt in ("Fix add() in calc.py.", "Describe the project.")


def test_append_task_multiline(tmp_path):
    p = tmp_path / "s.toml"
    append_task(p, {"id": "a", "prompt": 'line one\nline "two"', "tags": ["x"]})
    append_task(p, {"id": "b", "prompt": "p"})
    s = load_suite(p)
    assert s.tasks[0].prompt == 'line one\nline "two"' and [t.id for t in s.tasks] == ["a", "b"]


def test_stats():
    lo, hi = wilson(5, 10)
    assert round(lo, 3) == 0.237 and round(hi, 3) == 0.763
    assert wilson(0, 0) == (0.0, 0.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == 1.0


def test_format_report_smoke():
    s = {
        "batch": "b1", "kind": "suite", "name": "n", "meta": {"tasks": ["t"], "repeat": 1},
        "variants": [{"label": "x", "config": "c", "runs": 1, "passes": 1, "pass_rate": 1.0, "ci95": [0.2, 1.0],
                      "mean_turns": 3, "mean_gen_tokens": 10, "mean_prompt_tokens": 100, "mean_max_context": 50,
                      "mean_wall_s": 1.0, "overhead_tokens": 300, "statuses": {"completed": 1}, "failure_runs": {}}],
        "tasks": {"t": {"x": [1, 1]}},
    }  # fmt: skip
    text = format_report(s)
    assert "x" in text and "1/1" in text and os.linesep.join([]) == ""
