"""Replay provider fixtures: every ``*.json`` under ``tests/fixtures/providers``.

Each fixture holds argus's canonical request, the wire request the adapter must
send, the response in the provider's own format, and what the adapter must
parse from it (see :mod:`argus.providers.record` for the format). Recordings
made with ``argus run --record DIR`` have the same shape; point
``ARGUS_FIXTURES`` at extra directories (``:``-separated) to replay them too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from argus.mock import MockServer, Script
from argus.providers import ProviderOptions, make_provider

HERE = Path(__file__).parent / "fixtures" / "providers"
BASE_PATHS = {"anthropic": "/v1", "openai": "/v1", "gemini": "/v1beta", "openrouter": "/api/v1"}


def fixture_files() -> list[Path]:
    dirs = [HERE] + [Path(d) for d in os.environ.get("ARGUS_FIXTURES", "").split(":") if d]
    return sorted(p for d in dirs if d.is_dir() for p in d.rglob("*.json"))


def _id(p: Path) -> str:
    try:
        return str(p.relative_to(HERE).with_suffix(""))
    except ValueError:
        return p.stem


def run_fixture(fx: dict):
    server = MockServer(
        Script([{"replay": fx["response"]}]),
        model=fx.get("model", "m"),
        verify_signatures=False,  # real signatures and encrypted reasoning are opaque
    ).start()
    try:
        kind = fx["provider"]
        base = server.root + BASE_PATHS.get(kind, "/v1")
        opts = ProviderOptions(
            flavor=fx.get("flavor") or "auto",
            quirks=frozenset(fx.get("quirks") or []),
            **{k: v for k, v in (fx.get("options") or {}).items()},
        )
        p = make_provider(
            kind, base_url=base, model=fx.get("model", ""), api_key="test-key", retries=0,
            options=opts,
        )  # fmt: skip
        try:
            result = p.chat(fx["canonical"], stream=fx["stream"])
            error = None
        except Exception as e:  # noqa: BLE001 - fixtures may expect any adapter error
            result, error = None, e
        finally:
            p.close()
        return server, result, error
    finally:
        server.stop()


def subset_equal(want, got, path="") -> list[str]:
    """Differences where ``want`` is not reproduced by ``got`` (exact for leaves)."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            if k not in got:
                out.append(f"{path}.{k}: missing")
            elif k not in want:
                out.append(f"{path}.{k}: unexpected {json.dumps(got[k])[:80]}")
            else:
                out += subset_equal(want[k], got[k], f"{path}.{k}")
        return out
    return [] if want == got else [f"{path}: want {want!r}, got {got!r}"]


@pytest.mark.parametrize("path", fixture_files(), ids=_id)
def test_fixture(path: Path):
    fx = json.loads(path.read_text())
    assert fx.get("format") == 1, "unknown fixture format"
    server, c, error = run_fixture(fx)

    (wire,) = server.requests
    assert subset_equal(fx["request"]["body"], wire) == []
    assert server.paths[0].endswith(fx["request"]["path"])

    if "expect_error" in fx or ("error" in fx and "expect" not in fx):
        want = fx.get("expect_error") or fx["error"]["type"]
        assert error is not None and type(error).__name__ == want, error
        return
    assert error is None, error
    exp = fx["expect"]
    got = {
        "content": c.content,
        "reasoning": c.reasoning,
        "tool_calls": [[t.id, t.name, t.arguments] for t in c.tool_calls],
        "finish_reason": c.finish_reason,
        "usage": c.usage,
        "replay": c.replay,
        "refusal": c.refusal,
        "aborted": c.aborted,
    }
    for key, value in exp.items():
        assert got[key] == value, f"{key}: want {value!r}, got {got[key]!r}"


def test_fixtures_exist_for_every_provider():
    kinds = {json.loads(p.read_text())["provider"] for p in fixture_files()}
    assert {"openai_compat", "openrouter", "anthropic", "openai", "gemini"} <= kinds
