"""Capture the numbers and terminal output shown in the promo from real argus runs.

Everything runs against argus's own mock server, never a real model:

* prompt overhead per protocol (``argus overhead --all --json``)
* a scripted ``argus run -v`` session and the ``argus show last`` that follows
* the example A/B report numbers, computed with argus's own statistics code

Writes ``promo/src/data.json``. Run from the repository root with the project venv:

    .venv/bin/python promo/scripts/capture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from argus.cli import main as argus_main
from argus.mock import MockServer, Script, call, final
from argus.report import mcnemar_exact, wilson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "promo" / "src" / "data.json"
DEMO_WS = Path("/tmp/demo/calc")  # shown in the transcript, so keep it short and stable

CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
TEST_CALC = (
    "from calc import add, mul\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_add_negative():\n    assert add(-1, 1) == 0\n\n\n"
    "def test_mul():\n    assert mul(2, 3) == 6\n"
)
TASK = "test_add fails; fix calc.py"
TEST_CMD = "python3 -m pytest -q 2>&1 | tail -1"


def run_cli(args: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = argus_main(args)
    if rc != 0:
        raise SystemExit(f"argus {' '.join(args)} failed ({rc})")
    return buf.getvalue()


def overhead(url: str, workdir: Path) -> dict:
    data = json.loads(
        run_cli(
            ["overhead", "--all", "--json", "-w", str(workdir), "-o", f'model.base_url="{url}"']
        )
    )
    return {
        name: {"total": ov["total"], "method": ov["method"], "call_cost": ov["call_cost"]}
        for name, ov in data.items()
    }


REASONING = [
    "test_add expects add(2, 3) == 5. Read calc.py first.",
    "add() subtracts. Swap the operator.",
    "Run the test suite to confirm.",
]
ANSWER = "Fixed add() in calc.py: it returned a - b instead of a + b. All 3 tests pass."


def demo_script() -> Script:
    return Script(
        [
            {"reasoning": REASONING[0], **call("read", path="calc.py")},
            {
                "reasoning": REASONING[1],
                **call("edit", path="calc.py", old="return a - b", new="return a + b"),
            },
            {"reasoning": REASONING[2], **call("bash", cmd=TEST_CMD)},
            final(ANSWER),
        ]
    )


def classify(line: str) -> str:
    """Line kind, used only for colouring in the video."""
    s = line.strip()
    if line in REASONING:
        return "reasoning"
    if line == ANSWER:
        return "answer"
    if re.match(r"^\[\d+\] ", line):
        return "turn"
    if line.startswith("run ") and ":" in line and "  " not in line:
        return "meta"
    if s.startswith("✓"):
        return "ok"
    if s.startswith(("✗", "!")):
        return "fail"
    if line.startswith("completed"):
        return "status"
    if re.match(r"^  \w+\(", line):
        return "tool"
    if line.startswith("── turn"):
        return "heading"
    if re.match(r"^(task|config|workspace|tokens|time|failures|batch|check):", line):
        return "field"
    if line.startswith("run ") or line == "final answer:":
        return "title"
    return "plain"


def lines(text: str) -> list[dict]:
    return [{"text": ln, "kind": classify(ln)} for ln in text.rstrip("\n").split("\n")]


CHECKPOINT_ROW = re.compile(r"^\s*(\d+)\s+t(\S+)\s+([0-9a-f]{10})\s+(.+?)\s{2,}(.*)$")


def checkpoints(env: dict) -> list[dict]:
    out = subprocess.run(
        [sys.executable, "-m", "argus", "checkpoints", "last"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for ln in out.splitlines()[1:]:
        m = CHECKPOINT_ROW.match(ln + "  ")
        if m:
            idx, turn, commit, reason, change = m.groups()
            rows.append(
                {"reason": reason.strip(), "commit": commit[:7], "change": change.strip() or ""}
            )
    return rows


def terminal(tmp: Path) -> dict:
    if DEMO_WS.exists():
        shutil.rmtree(DEMO_WS)
    DEMO_WS.mkdir(parents=True)
    (DEMO_WS / "calc.py").write_text(CALC)
    (DEMO_WS / "test_calc.py").write_text(TEST_CALC)
    home = tmp / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),  # no global AGENTS.md or skills
        "NO_COLOR": "1",
        "ARGUS_DB": str(tmp / "demo.db"),
        "ARGUS_SHADOW_ROOT": str(tmp / "shadow"),
        # the venv's python has pytest, which the scripted bash call runs
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    with MockServer(demo_script(), chunk_delay=0.012) as server:
        run = subprocess.run(
            [sys.executable, "-m", "argus", "run", TASK, "-v", "-w", str(DEMO_WS),
             "-o", f'model.base_url="{server.url}"'],
            env=env, capture_output=True, text=True, check=True,
        )  # fmt: skip
    show = subprocess.run(
        [sys.executable, "-m", "argus", "show", "last"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "run": {"command": f'argus run "{TASK}" -v', "output": lines(run.stderr + run.stdout)},
        "show": {"command": "argus show last", "output": lines(show.stdout)},
        "checkpoints": checkpoints(env),
    }


def ab_example() -> dict:
    """Illustrative A/B numbers, pushed through argus's own statistics."""
    runs, native, grammar, only_native, only_grammar = 16, 9, 15, 0, 6
    return {
        "illustrative": True,
        "runs": runs,
        "variants": [
            {"label": "native", "passes": native, "ci95": wilson(native, runs)},
            {"label": "grammar", "passes": grammar, "ci95": wilson(grammar, runs)},
        ],
        "only_a": only_native,
        "only_b": only_grammar,
        "mcnemar_p": mcnemar_exact(only_native, only_grammar),
    }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ws = tmp / "ws"
        ws.mkdir()
        (ws / "calc.py").write_text(CALC)
        with MockServer(Script([])) as server:
            ov = overhead(server.url, ws)
        data = {
            "source": "argus mock-server (no real model)",
            "argus_commit": commit.strip(),
            "overhead": ov,
            "terminal": terminal(tmp),
            "ab": ab_example(),
        }
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(json.dumps(data["overhead"], indent=2))


if __name__ == "__main__":
    main()
