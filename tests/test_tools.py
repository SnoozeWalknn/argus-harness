from __future__ import annotations

import pytest

from argus.config import ToolsConfig
from argus.executors import LocalExecutor
from argus.tools import ToolContext, ToolError, build_tools
from argus.tools.search import glob_regex


@pytest.fixture
def ctx(workspace):
    return ToolContext(LocalExecutor(str(workspace)), ToolsConfig())


@pytest.fixture
def tools():
    return build_tools(ToolsConfig())


# -- read ----------------------------------------------------------------------------------


def test_read_numbers_lines(ctx, tools):
    r = tools["read"].run(ctx, {"path": "calc.py"})
    assert r.text.splitlines()[0] == "1\tdef add(a, b):"
    assert r.text.splitlines()[1] == "2\t    return a - b"
    assert not r.truncated


def test_read_offset_limit(ctx, tools):
    r = tools["read"].run(ctx, {"path": "calc.py", "offset": 2, "limit": 2})
    lines = r.text.splitlines()
    assert lines[0].startswith("2\t") and lines[1].startswith("3\t")
    assert "[lines 2-3 of 6; offset=4 for more]" in r.text


def test_read_errors(ctx, tools, workspace):
    with pytest.raises(ToolError, match="did you mean: calc.py"):
        tools["read"].run(ctx, {"path": "calcs.py"})
    with pytest.raises(ToolError, match="did you mean: pkg/util.py"):
        tools["read"].run(ctx, {"path": "util.py"})
    with pytest.raises(ToolError, match="directory"):
        tools["read"].run(ctx, {"path": "pkg"})
    (workspace / "b.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        tools["read"].run(ctx, {"path": "b.bin"})
    with pytest.raises(ToolError, match="past the end"):
        tools["read"].run(ctx, {"path": "calc.py", "offset": 99})


def test_read_long_lines_clipped(ctx, tools, workspace):
    (workspace / "long.txt").write_text("x" * 1000 + "\n")
    r = tools["read"].run(ctx, {"path": "long.txt"})
    assert "…[+600 chars]" in r.text


# -- edit ----------------------------------------------------------------------------------


def test_edit_requires_read(ctx, tools):
    with pytest.raises(ToolError, match="read calc.py before editing"):
        tools["edit"].run(ctx, {"path": "calc.py", "old": "a - b", "new": "a + b"})


def test_edit_exact(ctx, tools, workspace):
    tools["read"].run(ctx, {"path": "calc.py"})
    r = tools["edit"].run(ctx, {"path": "calc.py", "old": "a - b", "new": "a + b"})
    assert "Edited calc.py (lines 2-2)" in r.text
    assert "2\t    return a + b" in r.text  # snippet
    assert "return a + b" in (workspace / "calc.py").read_text()


def test_edit_ambiguous_and_all(ctx, tools, workspace):
    tools["read"].run(ctx, {"path": "calc.py"})
    with pytest.raises(ToolError, match=r"matches 2 times in calc.py \(lines 1, 5\)"):
        tools["edit"].run(ctx, {"path": "calc.py", "old": "(a, b)", "new": "(x, y)"})
    r = tools["edit"].run(ctx, {"path": "calc.py", "old": "(a, b)", "new": "(x, y)", "all": True})
    assert "2 occurrences" in r.text
    assert (workspace / "calc.py").read_text().count("(x, y)") == 2


def test_edit_create_and_errors(ctx, tools, workspace):
    r = tools["edit"].run(ctx, {"path": "new/mod.py", "old": "", "new": "X = 1\n"})
    assert "Created new/mod.py" in r.text
    assert (workspace / "new/mod.py").read_text() == "X = 1\n"
    with pytest.raises(ToolError, match="already exists"):
        tools["edit"].run(ctx, {"path": "new/mod.py", "old": "", "new": "Y"})
    with pytest.raises(ToolError, match='does not exist; to create it use old=""'):
        tools["edit"].run(ctx, {"path": "nope.py", "old": "a", "new": "b"})
    with pytest.raises(ToolError, match="outside the workspace"):
        tools["edit"].run(ctx, {"path": "/tmp/argus-evil.py", "old": "", "new": "x"})
    with pytest.raises(ToolError, match="identical"):
        tools["read"].run(ctx, {"path": "calc.py"})
        tools["edit"].run(ctx, {"path": "calc.py", "old": "a", "new": "a"})


def test_edit_syntax_warning(ctx, tools):
    tools["read"].run(ctx, {"path": "calc.py"})
    r = tools["edit"].run(ctx, {"path": "calc.py", "old": "return a - b", "new": "return (a + b"})
    assert "syntax error" in r.text
    assert r.meta["syntax_warning"]


def test_edit_preserves_mode(ctx, tools, workspace):
    p = workspace / "run.sh"
    p.write_text("echo hi\n")
    p.chmod(0o755)
    tools["read"].run(ctx, {"path": "run.sh"})
    tools["edit"].run(ctx, {"path": "run.sh", "old": "hi", "new": "bye"})
    assert p.stat().st_mode & 0o777 == 0o755


# -- bash ----------------------------------------------------------------------------------


def test_bash_output_and_exit(ctx, tools):
    r = tools["bash"].run(ctx, {"cmd": "echo out; echo err >&2; exit 3"})
    assert "out" in r.text and "err" in r.text
    assert r.text.endswith("[exit 3]")
    assert r.meta["exit_code"] == 3
    r = tools["bash"].run(ctx, {"cmd": "true"})
    assert r.text == "[exit 0, no output]"


def test_bash_runs_in_workdir(ctx, tools, workspace):
    r = tools["bash"].run(ctx, {"cmd": "pwd"})
    assert str(workspace) in r.text


def test_bash_timeout_kills_group(ctx, tools):
    r = tools["bash"].run(ctx, {"cmd": "sleep 30 & sleep 30; echo never", "timeout": 1})
    assert "[timed out after 1s]" in r.text
    assert r.meta["timed_out"]


def test_bash_truncates(ctx, tools):
    r = tools["bash"].run(ctx, {"cmd": "seq 1 5000"})
    assert r.truncated
    assert "lines omitted" in r.text
    assert r.text.splitlines()[0] == "1"
    assert "5000" in r.text


def test_bash_no_stdin_hang(ctx, tools):
    r = tools["bash"].run(ctx, {"cmd": "cat; echo done", "timeout": 5})
    assert "done" in r.text


# -- glob / grep -----------------------------------------------------------------------------


def test_glob(ctx, tools):
    r = tools["glob"].run(ctx, {"pattern": "*.py"})
    assert r.text.splitlines() == ["calc.py", "pkg/__init__.py", "pkg/util.py", "test_calc.py"]
    r = tools["glob"].run(ctx, {"pattern": "pkg/*.py"})
    assert r.text.splitlines() == ["pkg/__init__.py", "pkg/util.py"]
    r = tools["glob"].run(ctx, {"pattern": "*.py", "path": "pkg"})
    assert r.text.splitlines() == ["pkg/__init__.py", "pkg/util.py"]
    r = tools["glob"].run(ctx, {"pattern": "**/*.{md,txt}"})
    assert r.text == "README.md"
    assert tools["glob"].run(ctx, {"pattern": "*.rs"}).text == "No files matched."


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("*.py", "a/b/c.py", True),
        ("src/*.py", "src/a.py", True),
        ("src/*.py", "src/x/a.py", False),
        ("src/**/*.py", "src/x/y/a.py", True),
        ("src/**/*.py", "src/a.py", True),
        ("**/test_*.py", "test_a.py", True),
        ("[ab].txt", "a.txt", True),
        ("[!ab].txt", "a.txt", False),
        ("*.{ts,tsx}", "x/y.tsx", True),
        ("./README.md", "README.md", True),
    ],
)
def test_glob_regex(pattern, path, expected):
    assert bool(glob_regex(pattern).match(path)) is expected


def test_grep(ctx, tools):
    r = tools["grep"].run(ctx, {"pattern": r"def \w+"})
    lines = r.text.splitlines()
    assert "calc.py:1:def add(a, b):" in lines
    assert any(line.startswith("pkg/util.py:1:") for line in lines)
    r = tools["grep"].run(ctx, {"pattern": "DEF ADD", "ignore_case": True, "glob": "*.py"})
    assert "calc.py:1:def add(a, b):" in r.text
    r = tools["grep"].run(ctx, {"pattern": "helper", "path": "pkg"})
    assert r.text.startswith("pkg/util.py:1:")
    assert tools["grep"].run(ctx, {"pattern": "zzzz_nothing"}).text == "No matches."


def test_grep_bad_regex(ctx, tools):
    with pytest.raises(ToolError):
        tools["grep"].run(ctx, {"pattern": "(unclosed"})


def test_grep_fallback_without_rg(ctx, tools):
    ctx.executor._commands["rg"] = False
    r = tools["grep"].run(ctx, {"pattern": "def add"})
    assert "calc.py:1:def add(a, b):" in r.text


def test_list_files_respects_gitignore(workspace):
    from tests.conftest import git_init

    (workspace / ".gitignore").write_text("ignored/\n")
    (workspace / "ignored").mkdir()
    (workspace / "ignored" / "x.py").write_text("")
    git_init(workspace)
    (workspace / "untracked.py").write_text("")
    (workspace / "README.md").unlink()
    files = LocalExecutor(str(workspace)).list_files()
    assert "untracked.py" in files
    assert "ignored/x.py" not in files
    assert "README.md" not in files  # deleted from the worktree
