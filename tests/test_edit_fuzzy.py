from __future__ import annotations

import pytest

from argus.config import ToolsConfig
from argus.executors import LocalExecutor
from argus.tools import ToolContext, ToolError, build_tools
from argus.tools.fuzzy import strip_line_numbers
from argus.tools.truncate import collapse_carriage_returns, fold_repeats, smart_truncate

SRC = """class Calc:
    def add(self, a, b):
        # add two numbers
        return a - b

    def mul(self, a, b):
        return a * b
"""


@pytest.fixture
def env(tmp_path):
    (tmp_path / "c.py").write_text(SRC)
    ctx = ToolContext(LocalExecutor(str(tmp_path)), ToolsConfig())
    tools = build_tools(ToolsConfig())
    tools["read"].run(ctx, {"path": "c.py"})

    def edit(old, new, **kw):
        return tools["edit"].run(ctx, {"path": "c.py", "old": old, "new": new, **kw})

    return tmp_path / "c.py", edit, ctx, tools


def test_copied_line_numbers(env):
    path, edit, *_ = env
    r = edit("4\t        return a - b", "4\t        return a + b")
    assert "ignored line-number prefixes" in r.text
    assert "        return a + b\n" in path.read_text()


def test_trailing_whitespace(env):
    path, edit, *_ = env
    r = edit(
        "    def add(self, a, b):   \n        # add two numbers  ",
        "    def add(self, a, b):\n        # sum",
    )
    assert "ignoring trailing whitespace" in r.text
    assert "# sum" in path.read_text()


def test_indentation_shift_reindents_new(env):
    path, edit, *_ = env
    old = "def mul(self, a, b):\n    return a * b"
    new = "def mul(self, a, b):\n    result = a * b\n    return result"
    r = edit(old, new)
    assert "ignoring indentation" in r.text
    text = path.read_text()
    assert "    def mul(self, a, b):\n        result = a * b\n        return result\n" in text


def test_near_miss_single_line_error(env):
    _, edit, *_ = env
    with pytest.raises(ToolError) as e:
        edit("# add two numbers \u2014 fast", "# x")
    msg = str(e.value)
    assert "Closest text (83% similar) is at lines 3-3" in msg
    assert "+        # add two numbers" in msg


def test_smart_quotes(tmp_path):
    (tmp_path / "q.py").write_text('msg = "it\'s here"\nother = 1\n')
    ctx = ToolContext(LocalExecutor(str(tmp_path)), ToolsConfig())
    tools = build_tools(ToolsConfig())
    tools["read"].run(ctx, {"path": "q.py"})
    r = tools["edit"].run(
        ctx, {"path": "q.py", "old": "msg = \u201cit\u2019s here\u201d", "new": 'msg = "ok"'}
    )
    assert "unicode" in r.text
    assert (tmp_path / "q.py").read_text().startswith('msg = "ok"\n')


def test_spacing_inside_line(env):
    path, edit, *_ = env
    r = edit("return a*b", "return b * a")
    assert "ignoring spacing" in r.text
    assert "return b * a" in path.read_text()


def test_similarity_applies_above_threshold(env):
    path, edit, *_ = env
    old = "    def add(self, a, b):\n        # add two number\n        return a - b"  # typo
    r = edit(old, "    def add(self, a, b):\n        return a + b")
    assert "similarity" in r.text
    assert "return a + b" in path.read_text() and "# add two numbers" not in path.read_text()


def test_similarity_disabled_gives_closest_and_diff(env):
    path, edit, ctx, _ = env
    ctx.cfg.fuzzy_threshold = 2.0
    with pytest.raises(ToolError) as e:
        edit("    def add(self, a, b):\n        # add 2 numbers\n        return a - b", "x")
    msg = str(e.value)
    assert "not found in c.py. Closest text (" in msg
    assert "at lines 2-4" in msg
    assert "3\t        # add two numbers" in msg
    assert "-        # add 2 numbers" in msg and "+        # add two numbers" in msg
    assert msg.endswith("Copy `old` exactly from the file, without line numbers.")
    assert path.read_text() == SRC


def test_no_similar_text(env):
    _, edit, *_ = env
    with pytest.raises(ToolError) as e:
        edit("completely unrelated text that is nowhere", "x")
    assert "Closest" not in str(e.value)


def test_ambiguous_fuzzy_match(tmp_path):
    (tmp_path / "d.py").write_text("if x:\n    go()\nelse:\n  go()\n")
    ctx = ToolContext(LocalExecutor(str(tmp_path)), ToolsConfig())
    tools = build_tools(ToolsConfig())
    tools["read"].run(ctx, {"path": "d.py"})
    with pytest.raises(
        ToolError, match=r"matches 2 places in d.py ignoring indentation \(lines 2, 4\)"
    ):
        tools["edit"].run(ctx, {"path": "d.py", "old": "go() ", "new": "stop()"})
    r = tools["edit"].run(ctx, {"path": "d.py", "old": "go() ", "new": "stop()", "all": True})
    assert "2 occurrences" in r.text
    assert (tmp_path / "d.py").read_text() == "if x:\n    stop()\nelse:\n  stop()\n"


def test_stale_file_hint(env):
    path, edit, *_ = env
    path.write_text(SRC.replace("# add two numbers", "# changed elsewhere"))
    with pytest.raises(ToolError, match="changed since you last read it"):
        edit("# add two numbers\n        return a - b\n    FOO", "x")


def test_crlf_preserved(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    ctx = ToolContext(LocalExecutor(str(tmp_path)), ToolsConfig())
    tools = build_tools(ToolsConfig())
    tools["read"].run(ctx, {"path": "w.txt"})
    tools["edit"].run(ctx, {"path": "w.txt", "old": "one\ntwo", "new": "1\n2"})
    assert (tmp_path / "w.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"


def test_strip_line_numbers():
    assert strip_line_numbers("12\tfoo\n13\t  bar") == "foo\n  bar"
    assert strip_line_numbers("  7| x") == "x"
    assert strip_line_numbers("12\tfoo\nbar") is None
    assert strip_line_numbers("x = 1") is None


# -- truncation -------------------------------------------------------------------------------


def test_carriage_returns_and_repeats():
    assert collapse_carriage_returns("a\r10%\r100%\nb\r\n") == "100%\nb\n"
    assert fold_repeats(["x"] * 5 + ["y"]) == [
        "x",
        "[... previous line repeated 4 more times]",
        "y",
    ]
    assert fold_repeats(["x", "x", "y"]) == ["x", "x", "y"]


def test_smart_truncate_keeps_errors_from_the_middle():
    lines = [f"test_{i} PASSED" for i in range(1000)]
    lines[500] = "test_500 FAILED - AssertionError: expected 5"
    text, truncated = smart_truncate("\n".join(lines), 3000, 10, 20, 200)
    assert truncated
    out = text.splitlines()
    assert out[0] == "test_0 PASSED" and out[-1] == "test_999 PASSED"
    assert "501: test_500 FAILED - AssertionError: expected 5" in out
    assert any("970 lines omitted; 1 notable lines kept" in ln for ln in out)
    assert len(text) <= 3000


def test_smart_truncate_small_output_only_cleaned():
    text, truncated = smart_truncate("\x1b[31mred\x1b[0m\n", 1000, 10, 10, 100)
    assert text == "red\n" and not truncated
    text, truncated = smart_truncate("same\n" * 50, 1000, 10, 10, 100)
    assert text.count("same") == 1 and truncated  # folded lines count as dropped


def test_smart_truncate_long_lines():
    text, truncated = smart_truncate("x" * 50_000, 2000, 10, 10, 300)
    assert truncated and len(text) <= 2000


def test_bash_reports_full_size(tmp_path):
    ctx = ToolContext(LocalExecutor(str(tmp_path)), ToolsConfig())
    r = build_tools(ToolsConfig())["bash"].run(ctx, {"cmd": "seq 1 20000"})
    assert "[output was" in r.text and "grep" in r.text
    assert r.text.endswith("[exit 0]")
