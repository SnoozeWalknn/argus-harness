"""Opinionated defaults that work out of the box (as in Omarchy).

* Config files are layered: ``~/.config/argus/config.toml`` (global), then the
  workspace's ``.argus/config.toml`` (project), then ``-c FILE``, then ``-o``.
* Without a configured model, argus looks for one: a local server on its usual
  port (llama-server, Ollama, LM Studio, vLLM), else the first cloud API whose key
  is set (Anthropic, OpenAI, Gemini, OpenRouter).
* ``argus init`` writes a commented global config; ``argus doctor`` reports what
  works on this machine.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

GLOBAL_DIR = "~/.config/argus"
LOCAL_SERVERS = [
    ("llama_server", "http://127.0.0.1:8080/v1"),
    ("ollama", "http://127.0.0.1:11434/v1"),
    ("lmstudio", "http://127.0.0.1:1234/v1"),
    ("vllm", "http://127.0.0.1:8000/v1"),
]
CLOUD_DEFAULTS = [
    ("ANTHROPIC_API_KEY", "anthropic", "claude-opus-5"),
    ("OPENAI_API_KEY", "openai", "gpt-5"),
    ("GEMINI_API_KEY", "gemini", "gemini-2.5-pro"),
    ("GOOGLE_API_KEY", "gemini", "gemini-2.5-pro"),
    ("OPENROUTER_API_KEY", "openrouter", "qwen/qwen3-coder"),
]
MODEL_KEYS = {"model.provider", "model.model", "model.base_url", "model.profile"}

DEFAULT_CONFIG = """\
# argus global configuration (argus init). Every key is optional; see
# https://github.com/SnoozeWalknn/argus-harness/blob/main/examples/argus.toml
# for all of them. A workspace can add .argus/config.toml; -c and -o win over both.

[model]
# Leave provider/model unset and argus picks: a local server (llama-server :8080,
# Ollama :11434, LM Studio :1234, vLLM :8000), else the first cloud API with a key
# in the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
# OPENROUTER_API_KEY). Or pin one:
# provider = "anthropic"
# model = "claude-opus-5"

[agent]
approval = "auto"      # read-only | ask | auto | full

[sandbox]
backend = "auto"       # bubblewrap if it works, else Landlock
network = true

[tui]
theme = "auto"         # auto follows Omarchy's current theme, else tokyo-night
"""


def global_dir() -> Path:
    return Path(os.path.expanduser(os.environ.get("ARGUS_CONFIG_DIR") or GLOBAL_DIR))


def config_layers(workdir: str | None, explicit: str | None) -> list[Path]:
    """Config files to merge, lowest precedence first."""
    layers = []
    g = global_dir() / "config.toml"
    if g.is_file():
        layers.append(g)
    if workdir:
        p = Path(workdir) / ".argus" / "config.toml"
        if p.is_file() and p.resolve() != g.resolve():
            layers.append(p)
    if explicit:
        layers.append(Path(explicit))
    return layers


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_layers(paths: list[Path]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for p in paths:
        try:
            layer = tomllib.loads(p.read_text())
        except FileNotFoundError:
            from argus.config import ConfigError

            raise ConfigError(f"config file not found: {p}") from None
        except tomllib.TOMLDecodeError as e:
            from argus.config import ConfigError

            raise ConfigError(f"{p}: {e}") from None
        data = deep_merge(data, layer)
    return data


def local_servers(
    timeout: float = 0.3, servers: list[tuple[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Local model servers that answer, with the models they serve."""
    from argus.providers import make_provider

    found = []
    for flavor, url in servers if servers is not None else _servers_from_env():
        p = make_provider(flavor, base_url=url, connect_timeout=timeout, retries=0)
        try:
            d = p.detect()
        finally:
            p.close()
        if d.models or d.context_window:
            found.append(
                {
                    "flavor": flavor,
                    "url": url,
                    "models": d.models,
                    "ctx": d.context_window,
                    "notes": d.notes,
                }
            )
    return found


def _servers_from_env() -> list[tuple[str, str]]:
    raw = os.environ.get("ARGUS_LOCAL_SERVERS")  # flavor=url,flavor=url; "" = none
    if raw is None:
        return LOCAL_SERVERS
    out = []
    for item in raw.split(","):
        flavor, _, url = item.partition("=")
        if flavor and url:
            out.append((flavor.strip(), url.strip()))
    return out


def pick_model(explicit: set[str]) -> tuple[list[str], str]:
    """Overrides choosing a model when none is configured, and a sentence saying why."""
    if explicit & MODEL_KEYS:
        return [], ""
    for server in local_servers():
        overrides = [f'model.provider="{server["flavor"]}"', f'model.base_url="{server["url"]}"']
        model = next((m for m in server["models"] if m), "")
        if model:
            overrides.append(f"model.model={json.dumps(model)}")
        return overrides, f"using {model or 'the model'} on {server['flavor']} at {server['url']}"
    for env, provider, model in CLOUD_DEFAULTS:
        if os.environ.get(env):
            return (
                [f'model.provider="{provider}"', f'model.model="{model}"'],
                f"using {provider}/{model} ({env} is set)",
            )
    return [], ""


def init(force: bool = False) -> list[str]:
    """Write the global config and create the global directories; returns what was done."""
    d = global_dir()
    done = []
    for sub in ("skills", "agents", "hooks"):
        p = d / sub
        if not p.exists():
            p.mkdir(parents=True)
            done.append(f"created {p}")
    cfg = d / "config.toml"
    if force or not cfg.exists():
        cfg.write_text(DEFAULT_CONFIG)
        done.append(f"wrote {cfg}")
    else:
        done.append(f"kept {cfg}")
    return done


def doctor() -> dict[str, Any]:
    """What works here: for ``argus doctor``."""
    from argus import __version__
    from argus.lsp import LANGUAGES
    from argus.providers import KEY_ENV
    from argus.sandbox import status

    report: dict[str, Any] = {
        "argus": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "config": [str(p) for p in config_layers(os.getcwd(), None)],
        "db": os.path.expanduser(os.environ.get("ARGUS_DB") or "~/.local/share/argus/argus.db"),
        "tools": {t: bool(shutil.which(t)) for t in ("git", "rg", "bwrap")},
        "sandbox": status(),
        "api_keys": sorted({n for names in KEY_ENV.values() for n in names if os.environ.get(n)}),
        "local_servers": local_servers(),
        "lsp": {
            lang: next((s[0] for s in spec["servers"] if shutil.which(s[0])), None)
            for lang, spec in LANGUAGES.items()
        },
    }
    try:
        import textual

        report["tui"] = f"textual {textual.__version__}"
    except ImportError:
        report["tui"] = None
    return report


def format_doctor(r: dict[str, Any]) -> str:
    ok, no = "✓", "✗"

    def row(label: str, value: str) -> str:
        return f"{label + ':':<16}{value}"

    sb = r["sandbox"]
    lines = [
        f"argus {r['argus']} · Python {r['python']} · {r['platform']}",
        row("config", ", ".join(r["config"]) or "built-in defaults (argus init writes one)"),
        row("log", r["db"]),
        row(
            "sandbox",
            f"{ok if sb['auto'] != 'none' else no} {sb['auto']}  (bwrap "
            f"{'works' if sb['bwrap_works'] else 'no'}, Landlock ABI {sb['landlock_abi']})",
        ),
    ]
    lines += [row(tool, ok if found else no) for tool, found in r["tools"].items()]
    lines.append(
        row("tui", f"{ok} {r['tui']}" if r["tui"] else f"{no} pip install 'argus-harness[tui]'")
    )
    lines.append(row("api keys", ", ".join(r["api_keys"]) or "none set"))
    if r["local_servers"]:
        for s in r["local_servers"]:
            models = ", ".join(s["models"][:3])
            lines.append(
                row("model server", f"{ok} {s['flavor']} {s['url']} ctx {s['ctx'] or '?'} {models}")
            )
            lines += [row("", f"note: {n}") for n in s["notes"]]
    else:
        lines.append(row("model server", f"{no} none on the usual ports"))
    for lang, server in r["lsp"].items():
        lines.append(row(f"lsp {lang}", f"{ok} {server}" if server else no))
    hints = hints_for(r)
    if hints:
        lines += ["", "suggestions:"] + [f"  {h}" for h in hints]
    return "\n".join(lines)


def hints_for(r: dict[str, Any]) -> list[str]:
    pm = package_manager()
    out = []
    if r["sandbox"]["auto"] == "none" and sys.platform.startswith("linux"):
        out.append(f"install bubblewrap for sandboxed commands: {pm.format('bubblewrap')}")
    if not r["tools"]["rg"]:
        out.append(f"install ripgrep for faster search: {pm.format('ripgrep')}")
    if not any(r["lsp"].values()):
        out.append(
            "install a language server for diagnostics after edits, e.g. "
            "`npm i -g pyright` or `uv tool install python-lsp-server`"
        )
    if not r["api_keys"] and not r["local_servers"]:
        out.append(
            "start a local model server (llama-server, Ollama, ...) or set an API key "
            "such as ANTHROPIC_API_KEY"
        )
    return out


def package_manager() -> str:
    """A package install command template for this system ('{}' = package)."""
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            k, _, v = line.partition("=")
            osr[k] = v.strip('"')
    except OSError:
        pass
    ids = f"{osr.get('ID', '')} {osr.get('ID_LIKE', '')}"
    if sys.platform == "darwin":
        return "brew install {}"
    if "arch" in ids:
        return "sudo pacman -S {}"
    if "debian" in ids or "ubuntu" in ids:
        return "sudo apt install {}"
    if "fedora" in ids or "rhel" in ids:
        return "sudo dnf install {}"
    if "suse" in ids:
        return "sudo zypper install {}"
    return "install {} with your package manager"
