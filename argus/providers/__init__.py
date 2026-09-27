"""Model API adapters.

argus builds every request in one canonical shape, the OpenAI chat-completions
format it also stores in SQLite, plus two neutral knobs:

``thinking``  ``{"enabled": bool | None, "effort": str, "budget": int}``
``cache``     ``bool``: ask for prompt caching where the API needs it requested

Assistant messages may carry ``replay``, opaque provider state (thinking
signatures, encrypted reasoning) that must be sent back verbatim. Each adapter
translates this to its wire format and back; see ``docs/MODEL-AGNOSTIC.md``.

Provider names: ``openai_compat`` (flavour auto-detected) and its flavours
``llama_server``, ``ollama``, ``vllm``, ``lmstudio``, ``openrouter``; ``anthropic``;
``openai`` (Responses API); ``gemini``.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from argus.providers.base import (
    Completion,
    ContextOverflow,
    Delta,
    Detected,
    LLMError,
    Monitor,
    NotSupported,
    Provider,
    ProviderOptions,
    RawToolCall,
    StopGeneration,
    root_url,
)

if TYPE_CHECKING:
    from argus.providers.record import Recorder

__all__ = [
    "Completion",
    "ContextOverflow",
    "Delta",
    "Detected",
    "LLMError",
    "Monitor",
    "NotSupported",
    "Provider",
    "ProviderOptions",
    "RawToolCall",
    "StopGeneration",
    "api_key_for",
    "make_provider",
    "parse_spec",
    "resolve_kind",
    "root_url",
    "strip_replay",
]

FLAVORS = ("llama_server", "ollama", "vllm", "lmstudio", "openrouter")
ADAPTERS = {
    "openai_compat": ("argus.providers.openai_compat", "OpenAICompatProvider"),
    "anthropic": ("argus.providers.anthropic", "AnthropicProvider"),
    "openai": ("argus.providers.openai", "OpenAIProvider"),
    "gemini": ("argus.providers.gemini", "GeminiProvider"),
}
PROVIDERS = ("auto", "openai_compat", *FLAVORS, "anthropic", "openai", "gemini")
KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}
HOSTS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "generativelanguage.googleapis.com": "gemini",
    "openrouter.ai": "openrouter",
}
# Spec prefixes: ``anthropic/claude-opus-5``; "llama" and "local" mean the llama-server.
SPEC_PREFIXES = {**{p: p for p in PROVIDERS if p != "auto"}, "llama": "llama_server"}


def resolve_kind(provider: str, base_url: str = "") -> str:
    """The provider name for a config: ``auto`` is decided by the base URL's host."""
    if provider and provider != "auto":
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
        return provider
    host = urlsplit(base_url).hostname or ""
    for suffix, kind in HOSTS.items():
        if host == suffix or host.endswith("." + suffix):
            return kind
    return "openai_compat"


def parse_spec(spec: str) -> tuple[str | None, str]:
    """``provider/model`` → (provider, model); a bare name or unknown prefix → (None, spec)."""
    head, sep, rest = spec.partition("/")
    if sep and head in SPEC_PREFIXES:
        return SPEC_PREFIXES[head], rest
    if spec in SPEC_PREFIXES and spec not in ("openai", "anthropic", "gemini"):
        return SPEC_PREFIXES[spec], ""
    return None, spec


def api_key_for(kind: str, explicit: str = "", env_name: str = "") -> str:
    """API key: explicit config value, else the named variable, else the provider's default."""
    if explicit:
        return explicit
    if env_name:
        return os.environ.get(env_name, "")
    for name in KEY_ENV.get(kind, ()):
        if os.environ.get(name):
            return os.environ[name]
    return ""


def adapter_class(kind: str) -> type[Provider]:
    key = "openai_compat" if kind in FLAVORS else kind
    module, name = ADAPTERS[key]
    return getattr(importlib.import_module(module), name)


def strip_replay(replay: dict[str, Any]) -> dict[str, Any] | None:
    """Replay state minus reasoning (see :meth:`Provider.strip_replay`)."""
    kind = replay.get("provider", "")
    if kind not in ADAPTERS:
        return None
    return adapter_class(kind).strip_replay(replay)


def make_provider(
    kind: str,
    *,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    api_key_env: str = "",
    timeout: float = 600.0,
    connect_timeout: float = 5.0,
    retries: int = 2,
    options: ProviderOptions | None = None,
    recorder: Recorder | None = None,
    require_key: bool = True,
) -> Provider:
    kind = resolve_kind(kind, base_url)
    opts = options or ProviderOptions()
    if kind in FLAVORS:
        opts.flavor = kind
    key = api_key_for("openrouter" if kind == "openrouter" else kind, api_key, api_key_env)
    needs_key = kind in ("anthropic", "openai", "gemini", "openrouter")
    if needs_key and not key and require_key:
        from argus.config import ConfigError

        names = api_key_env or " or ".join(KEY_ENV.get(kind, ()))
        raise ConfigError(f"{kind} needs an API key: set {names}")
    cls = adapter_class(kind)
    return cls(
        base_url,
        key,
        timeout,
        connect_timeout,
        retries,
        model=model,
        options=opts,
        recorder=recorder,
    )


def provider_from_config(m: Any, recorder: Recorder | None = None, **kw: Any) -> Provider:
    """Build the provider for a :class:`~argus.config.ModelConfig`."""
    opts = ProviderOptions(
        quirks=frozenset(m.quirks),
        effort=m.effort,
        thinking_budget=m.thinking_budget,
        cache=m.cache,
    )
    if recorder is None and m.record_dir:
        from argus.providers.record import Recorder

        recorder = Recorder(m.record_dir)
    if recorder is None:
        from argus.providers.record import recorder_from_env

        recorder = recorder_from_env()
    return make_provider(
        m.provider,
        base_url=m.base_url,
        model=m.model,
        api_key=m.api_key,
        api_key_env=m.api_key_env,
        timeout=m.timeout,
        connect_timeout=m.connect_timeout,
        retries=m.retries,
        options=opts,
        recorder=recorder,
        **kw,
    )
