"""Configuration: nested dataclasses loaded from TOML with dotted overrides.

Every key is optional. Unknown keys are rejected so typos fail loudly, and any
key can be overridden from the command line with ``-o section.key=value``.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    # auto (by base_url host) | openai_compat | llama_server | ollama | vllm | lmstudio |
    # openrouter | anthropic | openai | gemini
    provider: str = "auto"
    base_url: str = ""  # "" = the provider's default (llama-server: http://127.0.0.1:8080/v1)
    model: str = "qwen"
    profile: str = ""  # model profile to apply ("" = the best match for the model id)
    tier: str = ""  # frontier | local (from the profile); decides the "auto" tool set
    api_key: str = ""  # prefer api_key_env or the provider's standard variable; never logged
    api_key_env: str = ""  # environment variable holding the key (default: the provider's)
    # Sampling: None leaves the value to llama-server's own defaults.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    max_tokens: int = 8192  # per-turn generation cap
    context_window: int = 0  # 0 = ask the server (/props)
    stream: bool = True
    # None leaves thinking to the chat template; bool sets chat_template_kwargs.enable_thinking.
    enable_thinking: bool | None = None
    timeout: float = 600.0  # read timeout (prompt processing can be slow)
    connect_timeout: float = 5.0
    retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)
    thinking: str = ""  # thinking format for the adapter ("" = its default; see profiles)
    effort: str = ""  # reasoning effort where the API has one (low | medium | high | ...)
    thinking_budget: int = 0  # reasoning token budget where the API takes one
    cache: bool = True  # request prompt caching where the API needs it asked for
    quirks: list[str] = field(default_factory=list)  # named request adjustments, see profiles
    pricing: dict[str, float] = field(default_factory=dict)  # USD per 1M: input, output, cache_*
    record_dir: str = ""  # save every API exchange here as a fixture (also $ARGUS_RECORD)
    # Gemini's safety filter, for every adjustable category: "" = the API's defaults |
    # off | block_none | block_only_high | block_medium_and_above | block_low_and_above
    safety: str = ""


@dataclass
class AgentConfig:
    protocol: str = "native"  # native | json_schema | grammar
    max_turns: int = 50
    max_wall_seconds: float = 0  # 0 = no wall-clock limit (checked between turns)
    token_budget: int = 0  # cumulative completion tokens per run, 0 = unlimited
    max_reasoning_tokens: int = 0  # abort a turn's reasoning after this many tokens, 0 = off
    system_prompt: str = ""  # override; {workdir} is substituted
    keep_reasoning: bool = False  # send past reasoning back to the model
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    json_thought: bool = False  # constrained protocols: add a leading "thought" field
    grammar_think: bool = True  # grammar protocol: allow a <think>...</think> prefix
    loop_repeat: int = 3  # identical calls before a loop is tagged (and the model nudged)
    loop_abort: int = 5  # identical calls before the run is aborted
    max_malformed: int = 3  # consecutive malformed turns before aborting
    repetition_window: int = 400  # chars examined for degenerate repetition, 0 = off
    overrun_turns: int = 2  # tool turns after a completion claim before tagging overrun
    reasoning_retry: bool = True  # after a reasoning-budget abort, retry once with thinking off
    mode: str = "build"  # build | plan (read-only exploration that ends in a plan)
    subagents: str = "auto"  # the task tool: auto (frontier models) | on | off
    max_stop_blocks: int = 3  # times stop hooks may send the model back to work
    approval: str = "auto"  # read-only | ask | auto | full (see argus.approval)
    headless_approval: str = "deny"  # answer to "ask" when nobody can be asked: deny | allow


@dataclass
class ToolsConfig:
    enabled: list[str] = field(default_factory=lambda: ["read", "edit", "bash", "glob", "grep"])
    todo: str = "auto"  # the todo tool: auto (frontier models) | on | off
    descriptions: dict[str, str] = field(default_factory=dict)  # per-tool description overrides
    read_max_lines: int = 400
    read_max_line_chars: int = 400
    require_read_before_edit: bool = True
    allow_write_outside_workdir: bool = False
    fuzzy_threshold: float = 0.95  # similarity needed to apply a non-exact edit, >1 disables
    bash_timeout: float = 120.0
    bash_max_chars: int = 6000
    bash_head_lines: int = 40
    bash_tail_lines: int = 80
    capture_bytes: int = 256_000  # raw bytes kept from each end of command output
    grep_max_matches: int = 100
    glob_max_results: int = 200
    max_line_chars: int = 300  # grep/bash per-line cap


@dataclass
class ExecutorConfig:
    kind: str = "local"  # local | ssh
    workdir: str = ""  # default: current directory (local); required for ssh
    host: str = ""
    ssh_command: list[str] = field(default_factory=lambda: ["ssh"])
    ssh_options: list[str] = field(default_factory=list)
    control_persist: str = "10m"
    shell: str = "bash"
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    backend: str = "auto"  # auto | bwrap | landlock | none
    network: bool = True  # allow network in the workspace-write sandbox
    writable: list[str] = field(default_factory=list)  # extra writable paths, e.g. "~/.cache"
    required: bool = False  # refuse to run commands when no sandbox is available


@dataclass
class CheckpointConfig:
    enabled: bool = True
    shadow_root: str = ""  # default: ~/.local/share/argus/shadow on the executor host
    excludes: list[str] = field(
        default_factory=lambda: [
            "node_modules/",
            ".venv/",
            "venv/",
            "__pycache__/",
            "*.pyc",
            ".mypy_cache/",
            ".pytest_cache/",
            ".ruff_cache/",
            "target/",
            "dist/",
            "build/",
        ]
    )


@dataclass
class ContextConfig:
    agents_md: bool = True
    agents_md_names: list[str] = field(default_factory=lambda: ["AGENTS.md"])
    global_dir: str = "~/.config/argus"
    max_agents_md_chars: int = 8000
    skills: bool = True
    skill_dirs: list[str] = field(
        default_factory=lambda: [".agents/skills", ".claude/skills", "~/.config/argus/skills"]
    )
    agent_dirs: list[str] = field(  # subagent definitions (*.md)
        default_factory=lambda: [".argus/agents", ".claude/agents", "~/.config/argus/agents"]
    )


@dataclass
class CompactionConfig:
    enabled: bool = True
    threshold: float = 0.75  # fraction of the context window that triggers compaction
    keep_last_turns: int = 4  # recent turns never masked or summarised
    mask: bool = True  # stage 1: replace old tool outputs with stubs
    summarize: bool = True  # stage 2: summarise the middle with the small model
    provider: str = "auto"
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "qwen-small"
    api_key: str = ""
    api_key_env: str = ""
    max_tokens: int = 1024
    timeout: float = 120.0


@dataclass
class LSPConfig:
    enabled: bool = True  # diagnostics after edits, when a language server is installed
    servers: dict[str, list[str]] = field(default_factory=dict)  # language -> argv ([] = off)
    wait_ms: int = 3000  # how long to wait for the server's diagnostics after an edit
    warnings: bool = False  # report warnings too, not only errors
    max_items: int = 10
    timeout: float = 10.0  # per request (initialize gets at least 30s)


@dataclass
class TUIConfig:
    theme: str = "auto"  # auto: the last one picked, else Omarchy's current theme, else tokyo-night
    show_reasoning: bool = True


@dataclass
class HookConfig:
    event: str = ""  # session_start | user_prompt | pre_tool | post_tool | stop
    command: str = ""
    matcher: str = ""  # regex on the tool name for pre_tool / post_tool ("" = all)
    timeout: float = 30.0


@dataclass
class LogConfig:
    db: str = ""  # default: $ARGUS_DB or ~/.local/share/argus/argus.db
    measure_overhead: bool = True  # record prompt overhead tokens on every run


@dataclass
class Config:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    log: LogConfig = field(default_factory=LogConfig)
    lsp: LSPConfig = field(default_factory=LSPConfig)
    tui: TUIConfig = field(default_factory=TUIConfig)
    hooks: list[HookConfig] = field(default_factory=list)

    @property
    def explicit(self) -> set[str]:
        """Dotted keys set by a config file or an override (not defaults or profiles)."""
        if not hasattr(self, "_explicit"):
            object.__setattr__(self, "_explicit", set())
        return self._explicit  # type: ignore[attr-defined]

    def to_dict(self, redact: bool = False) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if redact:
            for section in ("model", "compaction"):
                if d[section].get("api_key"):
                    d[section]["api_key"] = "<redacted>"
        return d

    def hash(self) -> str:
        """Stable hash of everything except the display name and secrets."""
        d = self.to_dict(redact=True)
        d.pop("name", None)
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def db_path(self) -> Path:
        raw = self.log.db or os.environ.get("ARGUS_DB") or "~/.local/share/argus/argus.db"
        return Path(os.path.expanduser(raw))


PROTOCOLS = ("native", "json_schema", "grammar")
EXECUTORS = ("local", "ssh")


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(value, a, where)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[0] if errors else f"{where}: invalid value {value!r}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_coerce(v, args[0], f"{where}[]") for v in value] if args else list(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table, got {value!r}")
        return dict(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table, got {value!r}")
        return _build(tp, value, where)
    return value


def _build(cls: type, data: dict[str, Any], where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        prefix = f"{where}." if where else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(prefix + k for k in sorted(unknown))}"
        )
    obj = cls()
    for key, value in data.items():
        path = f"{where}.{key}" if where else key
        setattr(obj, key, _coerce(value, hints[key], path))
    return obj


def _parse_scalar(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_override(cfg: Config, override: str) -> None:
    """Apply ``section.key=value`` (value parsed as TOML, else taken as a string)."""
    if "=" not in override:
        raise ConfigError(f"override must look like section.key=value: {override!r}")
    dotted, raw = override.split("=", 1)
    parts = dotted.strip().split(".")
    target: Any = cfg
    for p in parts[:-1]:
        if not dataclasses.is_dataclass(target) or not hasattr(target, p):
            raise ConfigError(f"unknown config key: {dotted}")
        target = getattr(target, p)
    key = parts[-1]
    if not dataclasses.is_dataclass(target) or key not in {
        f.name for f in dataclasses.fields(target)
    }:
        raise ConfigError(f"unknown config key: {dotted}")
    hint = typing.get_type_hints(type(target))[key]
    value = _parse_scalar(raw.strip())
    # Allow bare strings for str-typed keys even when they parse as something else.
    if hint is str and not isinstance(value, str):
        value = raw.strip()
    setattr(target, key, _coerce(value, hint, dotted))
    if isinstance(cfg, Config):
        cfg.explicit.add(dotted.strip())


def validate(cfg: Config) -> Config:
    from argus.providers import PROVIDERS

    for where, value in (("model", cfg.model.provider), ("compaction", cfg.compaction.provider)):
        if value not in PROVIDERS:
            raise ConfigError(f"{where}.provider must be one of {PROVIDERS}, got {value!r}")
    if cfg.agent.protocol not in PROTOCOLS:
        raise ConfigError(f"agent.protocol must be one of {PROTOCOLS}, got {cfg.agent.protocol!r}")
    from argus.providers.gemini import SAFETY_THRESHOLDS

    if cfg.model.safety and cfg.model.safety not in SAFETY_THRESHOLDS:
        raise ConfigError(
            f"model.safety must be one of {tuple(SAFETY_THRESHOLDS)}, got {cfg.model.safety!r}"
        )
    from argus.approval import POLICIES
    from argus.sandbox import BACKENDS

    if cfg.agent.approval not in POLICIES:
        raise ConfigError(f"agent.approval must be one of {POLICIES}, got {cfg.agent.approval!r}")
    if cfg.agent.mode not in ("build", "plan"):
        raise ConfigError(f"agent.mode must be build or plan, got {cfg.agent.mode!r}")
    for where, value in (("tools.todo", cfg.tools.todo), ("agent.subagents", cfg.agent.subagents)):
        if value not in ("auto", "on", "off"):
            raise ConfigError(f"{where} must be auto, on or off, got {value!r}")
    from argus.hooks import EVENTS

    for h in cfg.hooks:
        if h.event not in EVENTS or not h.command:
            raise ConfigError(f"each [[hooks]] needs an event from {EVENTS} and a command")
    if cfg.agent.headless_approval not in ("deny", "allow"):
        raise ConfigError("agent.headless_approval must be deny or allow")
    if cfg.sandbox.backend not in BACKENDS:
        raise ConfigError(f"sandbox.backend must be one of {BACKENDS}, got {cfg.sandbox.backend!r}")
    if cfg.executor.kind not in EXECUTORS:
        raise ConfigError(f"executor.kind must be one of {EXECUTORS}, got {cfg.executor.kind!r}")
    if cfg.executor.kind == "ssh" and not (cfg.executor.host and cfg.executor.workdir):
        raise ConfigError("executor.kind = 'ssh' needs executor.host and executor.workdir")
    if not 0 < cfg.compaction.threshold <= 1:
        raise ConfigError("compaction.threshold must be in (0, 1]")
    return cfg


def config_from_dict(data: dict[str, Any]) -> Config:
    """Rebuild a config stored with a run (``runs.config_json``)."""
    return _build(Config, data)


def explicit_keys(data: dict[str, Any]) -> set[str]:
    """Dotted keys a TOML config sets: ``{"model": {"base_url": …}}`` → ``model.base_url``."""
    sections = {f.name for f in dataclasses.fields(Config) if dataclasses.is_dataclass(f.type)}
    sections |= {
        f.name
        for f in dataclasses.fields(Config)
        if isinstance(f.type, str) and f.type.endswith("Config")
    }
    out = set()
    for key, value in data.items():
        if key in sections and isinstance(value, dict):
            out |= {f"{key}.{k}" for k in value}
        else:
            out.add(key)
    return out


def load_config(
    path: str | Path | list[Path] | None = None,
    overrides: list[str] | None = None,
    name: str | None = None,
    profiles: bool = True,
) -> Config:
    """Load a config file (or layers of them, lowest precedence first), apply overrides,
    then fill unset keys from the model's profile."""
    data: dict[str, Any] = {}
    paths = [Path(p) for p in path] if isinstance(path, list) else ([Path(path)] if path else [])
    if paths:
        from argus.defaults import read_layers

        data = read_layers(paths)
        data.setdefault("name", paths[-1].stem)
    cfg = _build(Config, data)
    cfg.explicit.update(explicit_keys(data) - {"name"})
    for o in overrides or []:
        apply_override(cfg, o)
    if name:
        cfg.name = name
    if profiles:
        from argus.profiles import ProfileError, apply_profile

        # kept so a subagent on another model can get that model's profile instead
        object.__setattr__(cfg, "unprofiled", copy.deepcopy(cfg))

        try:
            apply_profile(cfg)
        except ProfileError as e:
            raise ConfigError(str(e)) from None
    return validate(cfg)
