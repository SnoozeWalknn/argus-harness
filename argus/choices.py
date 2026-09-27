"""The models you can switch to: for the TUI's picker and ``argus models``.

Three groups, each entry a model spec that ``-m`` / ``/model`` accept:

* **recent**: models this log has run, newest first;
* **local**: models on local servers that answer right now (llama-server, Ollama,
  LM Studio, vLLM on their usual ports, or ``ARGUS_LOCAL_SERVERS``);
* **cloud**: the aliases (``opus``, ``gpt``, ``flash``…), marked ready when the
  provider's API key is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from argus.providers import FLAVORS, KEY_ENV, parse_spec


@dataclass
class Choice:
    spec: str  # what -m takes
    name: str  # short name shown first (alias or model)
    group: str  # recent | local | cloud
    ready: bool  # key set / server answering
    note: str = ""  # e.g. "needs ANTHROPIC_API_KEY"


def key_note(provider: str) -> tuple[bool, str]:
    names = KEY_ENV.get(provider)
    if not names:
        return True, ""
    if any(os.environ.get(n) for n in names):
        return True, ""
    return False, f"needs {' or '.join(names)}"


def recent(store: Any, limit: int = 8) -> list[Choice]:
    out: list[Choice] = []
    seen: set[str] = set()
    try:
        rows = store.q(
            "SELECT provider, model FROM runs WHERE model IS NOT NULL AND provider IS NOT NULL "
            "ORDER BY started_at DESC LIMIT 400"
        )
    except Exception:  # an old or empty log
        return out
    for row in rows:
        kind, _, url = (row["provider"] or "").partition(" ")
        model = row["model"] or ""
        if not kind or kind == "base":
            continue
        spec = f"{kind}/{model}" if model else kind
        if kind in FLAVORS or kind == "openai_compat":
            spec += f"@{url}" if url else ""
        if spec in seen:
            continue
        seen.add(spec)
        ready, note = key_note(kind)
        out.append(Choice(spec, model or kind, "recent", ready, note))
        if len(out) >= limit:
            break
    return out


def local(servers: list[dict[str, Any]] | None = None) -> list[Choice]:
    from argus.defaults import local_servers

    out = []
    for s in local_servers() if servers is None else servers:
        for model in s["models"] or [""]:
            spec = f"{s['flavor']}/{model}@{s['url']}" if model else f"{s['flavor']}@{s['url']}"
            ctx = f"ctx {s['ctx'] // 1000}k" if s.get("ctx") else ""
            out.append(
                Choice(spec, model or s["flavor"], "local", True, f"{s['flavor']} {ctx}".strip())
            )
    return out


def cloud() -> list[Choice]:
    from argus.profiles import load_aliases

    out = []
    for alias, target in load_aliases().items():
        provider, _ = parse_spec(target)
        if provider is None or (provider in FLAVORS and provider not in KEY_ENV):
            continue  # local servers are listed when they answer
        ready, note = key_note(provider)
        out.append(Choice(target, alias, "cloud", ready, note))
    return out


def model_choices(store: Any = None, probe: bool = True) -> list[Choice]:
    groups = (recent(store) if store is not None else []) + (local() if probe else []) + cloud()
    out, seen = [], set()
    for c in groups:
        if c.spec not in seen:
            seen.add(c.spec)
            out.append(c)
    return out
