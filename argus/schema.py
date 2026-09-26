"""A small JSON-schema subset: validation and lenient coercion of tool arguments.

Supported keywords: type, properties, required, additionalProperties, enum,
const, anyOf, oneOf, items, minimum, maximum, minLength, maxLength, minItems,
maxItems. That covers every schema argus generates.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, Any] = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _is_type(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(t)
    return (
        py is not None
        and isinstance(value, py)
        and not (t != "boolean" and isinstance(value, bool))
    )


def validate(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Return a list of human-readable problems (empty when valid)."""
    where = path or "value"
    errs: list[str] = []
    if "const" in schema and value != schema["const"]:
        return [f"{where} must be {schema['const']!r}"]
    if "enum" in schema and value not in schema["enum"]:
        opts = ", ".join(repr(v) for v in schema["enum"][:20])
        return [f"{where} must be one of {opts}; got {value!r}"]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = [validate(value, s, path) for s in schema[key]]
            ok = sum(1 for b in branches if not b)
            if ok == 0:
                best = min(branches, key=len)
                return best
            if key == "oneOf" and ok > 1:
                return [f"{where} matches more than one alternative"]
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(value, x) for x in types):
            return [f"{where} must be {' or '.join(types)}; got {type(value).__name__}"]
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"missing required {_join(path, req)}")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs.extend(validate(v, props[k], _join(path, k)))
            elif extra is False:
                allowed = ", ".join(props) or "none"
                errs.append(f"unexpected {_join(path, k)} (allowed: {allowed})")
            elif isinstance(extra, dict):
                errs.extend(validate(v, extra, _join(path, k)))
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{where} needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{where} allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs.extend(validate(v, schema["items"], f"{where}[{i}]"))
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{where} must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{where} must have at most {schema['maxLength']} characters")
    elif _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{where} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{where} must be <= {schema['maximum']}")
    return errs


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Fix harmless type slips small models make: "10" for 10, "true" for true, null optionals."""
    t = schema.get("type")
    if isinstance(value, dict) and (t == "object" or "properties" in schema):
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        out = {}
        for k, v in value.items():
            if v is None and k in props and k not in required and not _allows_null(props[k]):
                continue  # {"offset": null} means "not given"
            out[k] = coerce(v, props[k]) if k in props else v
        return out
    if isinstance(value, str):
        s = value.strip()
        if t == "integer":
            try:
                return int(s)
            except ValueError:
                return value
        if t == "number":
            try:
                return float(s)
            except ValueError:
                return value
        if t == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
    if t == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _allows_null(schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    return t == "null" or (isinstance(t, list) and "null" in t)
