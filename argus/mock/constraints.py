"""Check that constrained output honours the request's schema or grammar.

The real server makes violating output impossible; the mock checks its
scripted output instead, so a schema or grammar bug in argus fails the tests
rather than passing silently.
"""

from __future__ import annotations

import json
from typing import Any

from argus.schema import validate


def request_schema(req: dict[str, Any]) -> dict[str, Any] | None:
    rf = req.get("response_format") or {}
    if rf.get("type") == "json_schema":
        return (rf.get("json_schema") or {}).get("schema") or {}
    if rf.get("type") == "json_object":
        return rf.get("schema") or {}
    if req.get("json_schema"):
        return req["json_schema"]
    return None


def check(req: dict[str, Any], text: str) -> str | None:
    schema = request_schema(req)
    if schema is not None:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            return f"not valid JSON: {e}"
        errors = validate(value, schema)
        if errors:
            return "; ".join(errors)
    grammar = req.get("grammar")
    if grammar:
        from argus.mock.gbnf import Grammar, GrammarError

        try:
            if not Grammar(grammar).matches(text):
                return f"output does not match the GBNF grammar: {text[:200]!r}"
        except GrammarError as e:
            return f"invalid GBNF grammar: {e}"
    return None
