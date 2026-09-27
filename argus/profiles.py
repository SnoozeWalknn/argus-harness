"""Model profiles: what argus needs to know about a model beyond its id.

Profiles come from the built-in catalog (``argus/data/models.toml``) and the
user's ``~/.config/argus/models.toml`` (or ``$ARGUS_MODELS``); a user profile with
the same name overrides the built-in one field by field. The profile is chosen
by name (``model.profile`` or ``-m NAME``) or by the best ``match`` pattern for
the model id, and applied only to config keys the user did not set, so the
precedence is: catalog < user profile < config file < command line.

What a server can tell about itself (context window, pricing on OpenRouter)
is detected at run time by the provider and fills what is still unknown.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus.config import Config
    from argus.providers.base import Completion

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty")
PRICE_KEYS = ("input", "output", "cache_read", "cache_write")
STATIC_PROTOCOLS = {
    "anthropic": ("native",),
    "openai": ("native",),
    "gemini": ("native",),
    "ollama": ("native", "json_schema"),
    "lmstudio": ("native", "json_schema"),
    "openrouter": ("native", "json_schema"),
    "llama_server": ("native", "json_schema", "grammar"),
    "vllm": ("native", "json_schema", "grammar"),
}


@dataclass
class Profile:
    name: str
    extends: str = ""  # inherit every field not set here from this profile
    match: list[str] = field(default_factory=list)
    provider: str = ""
    base_url: str = ""
    model: str = ""
    context_window: int = 0
    max_output: int = 0
    protocol: str = ""
    thinking: str = ""
    effort: str = ""
    thinking_budget: int = 0
    enable_thinking: bool | None = None
    sampling: dict[str, float] = field(default_factory=dict)
    quirks: list[str] = field(default_factory=list)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    pricing: dict[str, float] = field(default_factory=dict)
    compaction_model: str = ""
    tier: str = ""  # frontier | local
    notes: str = ""
    tuned: dict[str, Any] = field(default_factory=dict)
    source: str = "builtin"  # builtin | user | builtin+user

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


FIELDS = {f.name for f in dataclasses.fields(Profile)} - {"name", "source"}


class ProfileError(ValueError):
    pass


def user_path() -> Path:
    raw = os.environ.get("ARGUS_MODELS") or "~/.config/argus/models.toml"
    return Path(os.path.expanduser(raw))


def _from_table(name: str, table: dict[str, Any], source: str, where: str) -> Profile:
    unknown = set(table) - FIELDS
    if unknown:
        raise ProfileError(f"{where}: unknown profile key(s) {', '.join(sorted(unknown))}")
    p = Profile(name=name, source=source)
    for key, value in table.items():
        setattr(p, key, value)
    return p


def _read(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text()).get("profiles") or {}
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"{path}: {e}") from None


def builtin_profiles() -> dict[str, Profile]:
    text = resources.files("argus.data").joinpath("models.toml").read_text()
    tables = tomllib.loads(text).get("profiles") or {}
    return {n: _from_table(n, t, "builtin", f"catalog [{n}]") for n, t in tables.items()}


def _builtin_tables() -> dict[str, dict[str, Any]]:
    text = resources.files("argus.data").joinpath("models.toml").read_text()
    return tomllib.loads(text).get("profiles") or {}


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    """Built-in and user profiles, with user fields overriding and ``extends`` resolved."""
    tables = {n: dict(t) for n, t in _builtin_tables().items()}
    sources = dict.fromkeys(tables, "builtin")
    path = path or user_path()
    for name, table in _read(path).items():
        if name in tables:
            tables[name] = {**tables[name], **table}
            sources[name] = "builtin+user"
        else:
            tables[name] = dict(table)
            sources[name] = "user"

    def resolved(name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
        table = tables[name]
        parent = table.get("extends")
        if not parent:
            return table
        if parent in seen or parent not in tables:
            why = "a cycle" if parent in seen else "an unknown profile"
            raise ProfileError(f"profile {name!r} extends {why}: {parent!r}")
        base = {k: v for k, v in resolved(parent, (*seen, name)).items() if k != "match"}
        return {**base, **table}

    out = {}
    for name in tables:
        where = f"{path if sources[name] != 'builtin' else 'catalog'} [profiles.{name}]"
        out[name] = _from_table(name, resolved(name), sources[name], where)
    return out


def _score(pattern: str, model: str) -> int:
    """How specifically ``pattern`` matches ``model`` (literal characters), or -1."""
    if not fnmatch.fnmatchcase(model.lower(), pattern.lower()):
        return -1
    return len(pattern) - pattern.count("*") - pattern.count("?")


def provider_fits(profile: Profile, provider: str) -> bool:
    if not profile.provider or provider in ("", "auto"):
        return True
    if provider == profile.provider:
        return True
    local = ("openai_compat", "llama_server", "ollama", "vllm", "lmstudio")
    return profile.provider in local and provider in local


def match(profiles: dict[str, Profile], model: str, provider: str = "") -> Profile | None:
    best, best_score = None, -1
    for p in profiles.values():
        if not provider_fits(p, provider):
            continue
        for pattern in p.match:
            s = _score(pattern, model)
            if s > best_score:
                best, best_score = p, s
    return best


def resolve(cfg: Config, profiles: dict[str, Profile] | None = None) -> Profile | None:
    profiles = profiles if profiles is not None else load_profiles()
    m = cfg.model
    if m.profile:
        if m.profile not in profiles:
            raise ProfileError(
                f"unknown model profile {m.profile!r}; see `argus models` for the list"
            )
        return profiles[m.profile]
    if m.model in profiles:  # -m NAME with a profile name
        return profiles[m.model]
    return match(profiles, m.model, m.provider)


def supported_protocols(provider: str) -> tuple[str, ...] | None:
    """Protocols a provider supports, when known without asking the server."""
    return STATIC_PROTOCOLS.get(provider)


def pick_protocol(wanted: str, provider: str) -> str:
    supported = supported_protocols(provider)
    if supported is None or wanted in supported:
        return wanted
    for fallback in ("json_schema", "native"):
        if fallback in supported:
            return fallback
    return "native"


def apply(cfg: Config, profile: Profile) -> dict[str, Any]:
    """Fill config keys the user did not set from the profile; returns what was applied."""
    explicit = cfg.explicit
    applied: dict[str, Any] = {}
    m, a, comp = cfg.model, cfg.agent, cfg.compaction

    def put(target: Any, section: str, key: str, value: Any) -> None:
        dotted = f"{section}.{key}"
        if dotted in explicit or value in (None, "", 0, [], {}):
            return
        setattr(target, key, value)
        applied[dotted] = value

    tuned = profile.tuned or {}
    put(m, "model", "profile", profile.name)
    if m.provider in ("", "auto"):
        put(m, "model", "provider", profile.provider)
    put(m, "model", "base_url", profile.base_url)
    if profile.model and m.model == profile.name:
        m.model = profile.model
        applied["model.model"] = profile.model
    put(m, "model", "context_window", profile.context_window)
    put(m, "model", "max_tokens", profile.max_output)
    put(m, "model", "thinking", profile.thinking)
    put(m, "model", "effort", profile.effort)
    put(m, "model", "thinking_budget", profile.thinking_budget)
    enable = tuned.get("enable_thinking", profile.enable_thinking)
    if enable is not None and "model.enable_thinking" not in explicit:
        m.enable_thinking = enable
        applied["model.enable_thinking"] = enable
    for key, value in (profile.sampling or {}).items():
        if key in SAMPLING_KEYS:
            put(m, "model", key, value)
    put(m, "model", "quirks", list(profile.quirks))
    put(m, "model", "pricing", dict(profile.pricing))
    if profile.chat_template_kwargs and "model.extra_body" not in explicit:
        kw = {**profile.chat_template_kwargs, **(m.extra_body.get("chat_template_kwargs") or {})}
        m.extra_body = {**m.extra_body, "chat_template_kwargs": kw}
        applied["model.extra_body"] = m.extra_body
    protocol = tuned.get("protocol") or profile.protocol
    if protocol:
        put(a, "agent", "protocol", pick_protocol(protocol, m.provider))
    if (
        profile.compaction_model
        and not {
            "compaction.model",
            "compaction.base_url",
            "compaction.provider",
        }
        & explicit
    ):
        # the small model lives on the same API as the main one
        comp.provider, comp.model = m.provider, profile.compaction_model
        comp.base_url, comp.api_key_env = m.base_url, m.api_key_env
        applied["compaction.model"] = f"{m.provider}/{profile.compaction_model}"
    return applied


def apply_profile(cfg: Config, profiles: dict[str, Profile] | None = None) -> Profile | None:
    profile = resolve(cfg, profiles)
    if profile is not None:
        cfg.profile_applied = apply(cfg, profile)
    return profile


# -- cost --------------------------------------------------------------------------------------


def turn_cost(c: Completion, pricing: dict[str, float]) -> float | None:
    """USD for one generation: the provider's own figure, else usage × pricing."""
    reported = c.usage.get("cost")
    if isinstance(reported, (int, float)):
        return float(reported)
    if not pricing or ("input" not in pricing and "output" not in pricing):
        return None
    inp = float(pricing.get("input", 0.0))
    cached, written = c.cached_tokens, c.cache_write_tokens
    fresh = max(c.prompt_tokens - cached - written, 0)
    usd = (
        fresh * inp
        + cached * float(pricing.get("cache_read", inp))
        + written * float(pricing.get("cache_write", inp))
        + c.completion_tokens * float(pricing.get("output", 0.0))
    )
    return usd / 1_000_000


# -- writing user profiles -----------------------------------------------------------------------


def _toml_key(k: str) -> str:
    return k if k.replace("-", "").replace("_", "").isalnum() else _toml_str(k)


def _toml_str(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{out}"'


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump_profiles(profiles: dict[str, dict[str, Any]]) -> str:
    """Serialise ``{name: table}`` as ``[profiles.name]`` sections (tables inline)."""
    lines = [
        "# argus model profiles (overrides for the built-in catalog).",
        "# `argus tune` rewrites this file; comments are not kept.",
        "",
    ]
    for name, table in profiles.items():
        lines.append(f"[profiles.{_toml_key(name)}]")
        tuned = table.get("tuned")
        for key, value in table.items():
            if key != "tuned" and value is not None:
                lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        if tuned:
            lines += ["", f"[profiles.{_toml_key(name)}.tuned]"]
            for key, value in tuned.items():
                lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def save_user_profile(name: str, updates: dict[str, Any], path: Path | None = None) -> Path:
    """Merge ``updates`` into the user's profile ``name`` and write the file."""
    path = path or user_path()
    tables = _read(path)
    table = dict(tables.get(name) or {})
    for key, value in updates.items():
        if key == "tuned":
            table["tuned"] = value
        else:
            table[key] = value
    tables[name] = table
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(dump_profiles(tables))
    tomllib.loads(tmp.read_text())  # never replace a readable file with an unreadable one
    tmp.replace(path)
    return path
