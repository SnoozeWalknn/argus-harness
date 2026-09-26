"""JSON schema → GBNF for llama-server's ``grammar`` parameter.

Unlike llama-server's own converter, the output here allows no insignificant
whitespace, so every constrained call is as short as possible, and keys are
emitted in schema property order (optional keys may be skipped).
Supports the subset argus generates: object (properties, required,
``additionalProperties: false``), array (items), string, integer, number,
boolean, null, enum, const, anyOf/oneOf.
"""

from __future__ import annotations

import json
import re
from typing import Any

PRIMITIVES = {
    "string": r'"\"" char* "\""',
    "char": r'[^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
    "hex": "[0-9a-fA-F]",
    "integer": r'"-"? ("0" | [1-9] [0-9]*)',
    "number": r'"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
    "boolean": r'"true" | "false"',
    "null": r'"null"',
    "value": "object | array | string | number | boolean | null",
    "object": r'"{" (string ":" value ("," string ":" value)*)? "}"',
    "array": r'"[" (value ("," value)*)? "]"',
}
PRIMITIVE_DEPS = {
    "string": ["char"],
    "char": ["hex"],
    "value": ["object", "array", "string", "number", "boolean", "null"],
    "object": ["string", "value"],
    "array": ["value"],
}

# Reasoning block that may precede the action. The opening tag is optional because
# some chat templates open <think> in the prompt. The body is any text not containing "</think>".
THINK_RULES = {
    "think": r'"<think>"? think-body "</think>" [ \t\n]*',
    "think-body": (
        r'([^<] | "<" ([^/] | "/" ([^t] | "t" ([^h] | "h" ([^i] | "i" ([^n] | "n" ([^k] | "k" [^>])))))))*'
    ),
}


def lit(s: str) -> str:
    """GBNF string literal."""
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def json_lit(value: Any) -> str:
    return lit(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


class GrammarBuilder:
    def __init__(self) -> None:
        self.rules: dict[str, str] = {}
        self._by_body: dict[str, str] = {}

    def _use(self, name: str) -> str:
        if name not in self.rules:
            self.rules[name] = PRIMITIVES[name]
            for dep in PRIMITIVE_DEPS.get(name, []):
                self._use(dep)
        return name

    def add(self, hint: str, body: str) -> str:
        if body in self._by_body:
            return self._by_body[body]
        name = re.sub(r"[^a-z0-9]+", "-", hint.lower()).strip("-") or "r"
        base, i = name, 2
        while name in self.rules or name in PRIMITIVES:
            name = f"{base}-{i}"
            i += 1
        self.rules[name] = body
        self._by_body[body] = name
        return name

    def visit(self, schema: dict[str, Any], hint: str) -> str:
        """Return a GBNF expression matching ``schema``."""
        if "const" in schema:
            return json_lit(schema["const"])
        if "enum" in schema:
            return "(" + " | ".join(json_lit(v) for v in schema["enum"]) + ")"
        for key in ("anyOf", "oneOf"):
            if key in schema:
                alts = [
                    self.visit(s, _branch_hint(s, f"{hint}-{i}")) for i, s in enumerate(schema[key])
                ]
                return self.add(hint, " | ".join(alts))
        t = schema.get("type")
        if isinstance(t, list):
            alts = [self.visit({**schema, "type": x}, f"{hint}-{x}") for x in t]
            return "(" + " | ".join(alts) + ")"
        if t == "object" or "properties" in schema:
            return self._object(schema, hint)
        if t == "array":
            items = schema.get("items")
            item = (
                self.visit(items, f"{hint}-item") if isinstance(items, dict) else self._use("value")
            )
            return self.add(hint, f'"[" ({item} ("," {item})*)? "]"')
        if t in ("string", "integer", "number", "boolean", "null"):
            return self._use(t)
        return self._use("value")

    def _object(self, schema: dict[str, Any], hint: str) -> str:
        props: dict[str, Any] = schema.get("properties", {})
        if not props:
            return self._use("object")
        keys = list(props)
        required = set(schema.get("required", []))
        kv = {k: f'{json_lit(k)} ":" {self.visit(props[k], f"{hint}-{k}")}' for k in keys}
        states: dict[tuple[int, bool], str | None] = {}

        def rest(i: int, comma: bool) -> str | None:
            """Keys ``keys[i:]`` in order, optional ones skippable; ``comma`` = something precedes."""
            if i == len(keys):
                return None
            if (i, comma) in states:
                return states[(i, comma)]
            k = keys[i]
            after = rest(i + 1, True)
            take = ('"," ' if comma else "") + kv[k] + (f" {after}" if after else "")
            if k in required:
                body = take
            else:
                skip = rest(i + 1, comma)
                body = f"({take} | {skip})" if skip else f"({take})?"
            name = self.add(f"{hint}-{i}{'c' if comma else ''}", body)
            states[(i, comma)] = name
            return name

        inner = rest(0, False)
        return self.add(hint, '"{" ' + (f"{inner} " if inner else "") + '"}"')

    def render(self, root: str) -> str:
        lines = [f"root ::= {root}"]
        lines += [f"{name} ::= {body}" for name, body in self.rules.items()]
        return "\n".join(lines) + "\n"


def _branch_hint(schema: dict[str, Any], default: str) -> str:
    """Name a union branch after its first string ``const`` property (e.g. the tool name)."""
    for prop in (schema.get("properties") or {}).values():
        if isinstance(prop, dict) and isinstance(prop.get("const"), str):
            return prop["const"]
    return default


def schema_to_gbnf(schema: dict[str, Any], think: bool = False) -> str:
    """Grammar for a JSON value matching ``schema``, optionally after a ``<think>`` block."""
    b = GrammarBuilder()
    value = b.visit(schema, "action")
    if think:
        b.rules.update(THINK_RULES)
        return b.render(f"think? {value}")
    return b.render(value)


def ordered(value: Any, schema: dict[str, Any]) -> Any:
    """Reorder object keys into schema property order, the order the grammar enforces."""
    for key in ("anyOf", "oneOf"):
        if key in schema:
            from argus.schema import validate

            for s in schema[key]:
                if not validate(value, s):
                    return ordered(value, s)
            return value
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        out = {k: ordered(value[k], props[k]) for k in props if k in value}
        out.update({k: v for k, v in value.items() if k not in out})
        return out
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [ordered(v, schema["items"]) for v in value]
    return value
