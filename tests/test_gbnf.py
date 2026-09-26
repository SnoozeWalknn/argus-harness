from __future__ import annotations

import json
import random

import pytest

from argus.config import ToolsConfig
from argus.gbnf import lit, ordered, schema_to_gbnf
from argus.mock.gbnf import Grammar, GrammarError
from argus.protocols import envelope_schema
from argus.schema import validate
from argus.tools import build_tools

# -- recogniser --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grammar,text,ok",
    [
        ('root ::= "a" | "b"', "a", True),
        ('root ::= "a" | "b"', "c", False),
        ('root ::= "a"*', "", True),
        ('root ::= "a"+', "", False),
        ('root ::= "ab"{2}', "abab", True),
        ('root ::= "ab"{2}', "ababab", False),
        ('root ::= "a"{1,3}', "aaa", True),
        ('root ::= "a"{1,3}', "aaaa", False),
        ('root ::= "a"{2,}', "aaaaa", True),
        ("root ::= [a-c]+ [^0-9]", "abcx", True),
        ("root ::= [a-c]+ [^0-9]", "abc5", False),
        (r'root ::= "\x41\n" [\t]', "A\n\t", True),
        ("root ::= x y\nx ::= [a-z]*\ny ::= [a-z]", "abc", True),  # ambiguous split
        ('root ::= (\n  "a" |\n  "b"\n)+  # comment', "abba", True),
        ("root ::= . .", "é!", True),
        ('root ::= "{" ws "}"\nws ::= [ \\t\\n]*', "{ \n }", True),
    ],
)
def test_recogniser(grammar, text, ok):
    assert Grammar(grammar).matches(text) is ok


def test_recogniser_errors():
    with pytest.raises(GrammarError, match="no root"):
        Grammar('x ::= "a"')
    with pytest.raises(GrammarError, match="undefined rule"):
        Grammar("root ::= missing")
    with pytest.raises(GrammarError):
        Grammar('root ::= "unterminated')


def test_recogniser_long_input():
    g = Grammar('root ::= "<" [^>]* ">"')
    assert g.matches("<" + "x" * 50_000 + ">")


# -- generator -----------------------------------------------------------------------------------


def test_lit_escapes():
    assert lit('a"b\\c\n') == r'"a\"b\\c\n"'
    assert Grammar(f"root ::= {lit(chr(1) + 'é')}").matches(chr(1) + "é")


TOOLS = build_tools(ToolsConfig())
STRINGS = [
    "",
    "x",
    'quote " and \\ backslash',
    "multi\nline\n\ttab",
    "unicode é ✓",
    "</think>",
    "{not json}",
]


def random_value(schema, rng):
    if "const" in schema:
        return schema["const"]
    t = schema.get("type")
    if t == "string":
        return rng.choice(STRINGS)
    if t == "integer":
        return rng.choice([0, 1, 42, -7, 10**9])
    if t == "boolean":
        return rng.random() < 0.5
    if t == "object":
        props = schema["properties"]
        return {
            k: random_value(v, rng)
            for k, v in props.items()
            if k in schema.get("required", []) or rng.random() < 0.5
        }
    raise AssertionError(schema)


@pytest.mark.parametrize("thought", [False, True])
def test_envelope_grammar_matches_schema(thought):
    """Every schema-valid action is accepted in compact form; perturbations are rejected."""
    rng = random.Random(7)
    schema = envelope_schema(TOOLS, thought)
    grammar = Grammar(schema_to_gbnf(schema, think=True))
    for _ in range(150):
        branch = rng.choice(schema["anyOf"])
        value = random_value(branch, rng)
        assert validate(value, schema) == []
        text = json.dumps(ordered(value, schema), separators=(",", ":"), ensure_ascii=False)
        assert grammar.matches(text), text
        assert grammar.matches(f"<think>plan {rng.random()}</think>\n{text}")
        # invalid variants
        bad = dict(value)
        bad["args"] = {**value["args"], "bogus": 1}
        assert not grammar.matches(json.dumps(bad, separators=(",", ":")))
        assert not grammar.matches(text[:-1])
    unknown = {"tool": "rm_rf", "args": {}}
    if thought:
        unknown = {"thought": "", **unknown}
    assert not grammar.matches(json.dumps(unknown, separators=(",", ":")))


def test_grammar_rejects_wrong_types_and_whitespace():
    g = Grammar(schema_to_gbnf(envelope_schema(TOOLS, False)))
    assert g.matches('{"tool":"read","args":{"path":"a","offset":3}}')
    assert not g.matches('{"tool":"read","args":{"path":"a","offset":"3"}}')
    assert not g.matches('{"tool":"read","args":{"path":"a","offset":3.5}}')
    assert not g.matches('{"tool":"read","args":{"offset":3}}')  # missing required
    assert not g.matches('{"tool":"read", "args":{"path":"a"}}')  # whitespace
    assert not g.matches('{"tool":"edit","args":{"path":"a","old":"x","new":"y","all":1}}')
    assert not g.matches('<think>x</think>{"tool":"read","args":{"path":"a"}}')  # think disabled


def test_optional_only_object_and_arrays():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "array", "items": {"enum": ["x", "y"]}},
            "c": {"type": ["string", "null"]},
        },
    }
    g = Grammar(schema_to_gbnf(schema))
    for text in ["{}", '{"a":1}', '{"b":["x","y"]}', '{"a":1,"c":null}', '{"c":"s"}', '{"b":[]}']:
        assert g.matches(text), text
    for text in ['{,"a":1}', '{"b":["z"]}', '{"c":1}', '{"b":["x"],"a":1}']:
        assert not g.matches(text), text


def test_generic_value_fallback():
    g = Grammar(schema_to_gbnf({"type": "object", "properties": {"v": {}}, "required": ["v"]}))
    assert g.matches('{"v":{"k":[1,2.5e3,true,null,"s"]}}')
    assert not g.matches('{"v":}')


def test_rule_names_are_llama_cpp_compatible():
    text = schema_to_gbnf(envelope_schema(TOOLS, True), think=True)
    for line in text.splitlines():
        name = line.split(" ::= ")[0]
        assert all(c.isalnum() or c == "-" for c in name), name
