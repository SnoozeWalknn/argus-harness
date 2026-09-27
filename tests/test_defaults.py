"""Out-of-the-box defaults: layered config, model auto-pick, init, doctor, the installer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from argus import __version__
from argus.cli import main
from argus.config import load_config
from argus.defaults import (
    DEFAULT_CONFIG,
    config_layers,
    doctor,
    format_doctor,
    init,
    local_servers,
    pick_model,
)
from argus.mock import final

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def global_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "global"
    monkeypatch.setenv("ARGUS_CONFIG_DIR", str(d))
    return d


# -- layered config ----------------------------------------------------------------------------------


def test_global_then_project_then_explicit_then_overrides(global_dir, workspace, tmp_path):
    global_dir.mkdir()
    (global_dir / "config.toml").write_text(
        '[model]\nmodel = "from-global"\n[agent]\napproval = "ask"\nmax_turns = 10\n'
    )
    (workspace / ".argus").mkdir()
    (workspace / ".argus/config.toml").write_text("[agent]\nmax_turns = 20\nmax_stop_blocks = 7\n")
    explicit = tmp_path / "x.toml"
    explicit.write_text('[model]\nmodel = "from-explicit"\n')

    layers = config_layers(str(workspace), str(explicit))
    assert layers == [
        global_dir / "config.toml",
        workspace / ".argus/config.toml",
        explicit,
    ]
    cfg = load_config(layers, ["agent.max_turns=30"])
    assert cfg.model.model == "from-explicit"
    assert cfg.agent.approval == "ask"  # only the global file sets it
    assert cfg.agent.max_stop_blocks == 7
    assert cfg.agent.max_turns == 30
    assert {"model.model", "agent.approval", "agent.max_turns"} <= cfg.explicit


def test_no_files_means_built_in_defaults(global_dir, workspace):
    assert config_layers(str(workspace), None) == []
    cfg = load_config(config_layers(str(workspace), None))
    assert cfg.agent.approval == "auto"


def test_missing_explicit_file_is_an_error(global_dir, workspace, capsys):
    rc = main(["run", "x", "-c", str(workspace / "nope.toml"), "-w", str(workspace)])
    assert rc == 2 and "not found" in capsys.readouterr().err


# -- choosing a model ----------------------------------------------------------------------------------


def test_a_local_server_wins(mock, monkeypatch):
    server = mock([], model="qwen3-coder-30b", n_ctx=65536)
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", f"llama_server={server.url}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    found = local_servers()
    assert [(s["flavor"], s["ctx"]) for s in found] == [("llama_server", 65536)]
    overrides, why = pick_model(set())
    assert 'model.provider="llama_server"' in overrides
    assert f'model.base_url="{server.url}"' in overrides
    assert 'model.model="qwen3-coder-30b.gguf"' in overrides  # llama-server names the file
    assert "qwen3-coder-30b.gguf on llama_server" in why


def test_a_server_that_does_not_answer_is_skipped(monkeypatch):
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", "ollama=http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-000000000000")
    assert local_servers() == []
    overrides, why = pick_model(set())
    assert overrides == ['model.provider="openai"', 'model.model="gpt-5"']
    assert "OPENAI_API_KEY is set" in why


def test_cloud_keys_in_order(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-0000")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza0000")
    assert pick_model(set())[0] == ['model.provider="gemini"', 'model.model="gemini-2.5-pro"']
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000")
    assert pick_model(set())[0] == ['model.provider="anthropic"', 'model.model="claude-opus-5"']


def test_nothing_found_and_explicit_choices_are_left_alone(monkeypatch):
    assert pick_model(set()) == ([], "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000")
    assert pick_model({"model.model"}) == ([], "")
    assert pick_model({"model.provider", "agent.approval"}) == ([], "")


def test_run_picks_the_local_server(mock, monkeypatch, workspace, tmp_path, capsys):
    server = mock([final("hello from the picked model")], model="picked-model")
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", f"llama_server={server.url}")
    rc = main(["run", "say hi", "-w", str(workspace), "--db", str(tmp_path / "a.db"), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "argus: using picked-model.gguf on llama_server" in captured.err
    assert json.loads(captured.out)["status"] == "completed"
    assert server.requests[0]["model"] == "picked-model.gguf"


def test_a_configured_model_is_not_overridden(mock, monkeypatch, workspace, tmp_path, capsys):
    configured = mock([final("configured")], model="configured-model")
    other = mock([], model="other")
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", f"llama_server={other.url}")
    (workspace / ".argus").mkdir()
    (workspace / ".argus/config.toml").write_text(f'[model]\nbase_url = "{configured.url}"\n')
    rc = main(["run", "hi", "-w", str(workspace), "--db", str(tmp_path / "a.db"), "-q"])
    assert rc == 0
    assert len(configured.requests) == 1 and other.requests == []


# -- init ------------------------------------------------------------------------------------------------


def test_init_writes_once_and_force_rewrites(global_dir):
    done = init()
    assert f"wrote {global_dir / 'config.toml'}" in done
    assert all((global_dir / d).is_dir() for d in ("skills", "agents", "hooks"))
    tomllib.loads((global_dir / "config.toml").read_text())  # valid TOML
    (global_dir / "config.toml").write_text("# mine\n")
    assert f"kept {global_dir / 'config.toml'}" in init()
    assert (global_dir / "config.toml").read_text() == "# mine\n"
    assert f"wrote {global_dir / 'config.toml'}" in init(force=True)
    assert (global_dir / "config.toml").read_text() == DEFAULT_CONFIG


def test_the_default_config_loads(global_dir, workspace):
    init()
    cfg = load_config(config_layers(str(workspace), None))
    assert (cfg.agent.approval, cfg.sandbox.backend, cfg.tui.theme) == ("auto", "auto", "auto")
    assert not cfg.explicit & {"model.provider", "model.model"}  # argus picks at run time


def test_cli_init_and_version(global_dir, capsys):
    assert main(["init"]) == 0
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"argus {__version__}"


# -- doctor ------------------------------------------------------------------------------------------------


def test_doctor_reports_servers_keys_and_sandbox(mock, monkeypatch, global_dir):
    server = mock([], model="qwen3-coder-30b", n_ctx=40960)
    monkeypatch.setenv("ARGUS_LOCAL_SERVERS", f"llama_server={server.url}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-0000000000")
    init()
    r = doctor()
    assert r["argus"] == __version__
    assert r["api_keys"] == ["ANTHROPIC_API_KEY"]
    assert r["local_servers"][0]["models"] == ["qwen3-coder-30b.gguf"]
    assert r["config"] == [str(global_dir / "config.toml")]
    assert set(r["sandbox"]) >= {"auto", "bwrap_works", "landlock_abi"}
    text = format_doctor(r)
    assert f"model server:   ✓ llama_server {server.url} ctx 40960 qwen3-coder-30b.gguf" in text
    assert "api keys:       ANTHROPIC_API_KEY" in text
    assert "sk-ant" not in text and "sk-ant" not in json.dumps(r)  # names only, never values


def test_doctor_suggests_what_is_missing(monkeypatch):
    r = doctor()
    r["tools"]["rg"] = False
    r["lsp"] = {k: None for k in r["lsp"]}
    r["sandbox"]["auto"] = "none"
    text = format_doctor(r)
    assert "model server:   ✗ none on the usual ports" in text
    assert "api keys:       none set" in text
    assert "suggestions:" in text
    assert "set an API key such as ANTHROPIC_API_KEY" in text
    assert "ripgrep" in text and "language server" in text
    if sys.platform.startswith("linux"):
        assert "bubblewrap" in text


def test_cli_doctor_json(capsys):
    assert main(["doctor", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["argus"] == __version__ and r["local_servers"] == []


# -- install.sh ----------------------------------------------------------------------------------------------

# The installer is tested with stand-ins for uv and pipx that "install" argus as a
# shim running this checkout, so the test needs no network. They log their argv.
FAKE_INSTALLER = """#!/bin/sh
echo "$(basename "$0") $*" >> "$FAKE_LOG"
if [ "$1 $2" = "tool dir" ]; then echo "$FAKE_BIN"; exit 0; fi
mkdir -p "$FAKE_BIN"
printf '#!/bin/sh\\nPYTHONPATH="%s" exec "%s" -m argus "$@"\\n' "$FAKE_ROOT" "$FAKE_PY" > "$FAKE_BIN/argus"
chmod +x "$FAKE_BIN/argus"
"""


def run_installer(tmp_path: Path, tool: str, **env: str) -> tuple[subprocess.CompletedProcess, str]:
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / tool).write_text(FAKE_INSTALLER)
    (tools / tool).chmod(0o755)
    path = [str(tools), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
    home = tmp_path / "home"
    full = {
        "PATH": os.pathsep.join(path),
        "HOME": str(home),
        "FAKE_LOG": str(tmp_path / "log"),
        "FAKE_BIN": str(tmp_path / "bin"),
        "FAKE_ROOT": str(ROOT),
        "FAKE_PY": sys.executable,
        "ARGUS_SOURCE": str(ROOT),
        "ARGUS_LOCAL_SERVERS": "",
        "ARGUS_LSP_AUTODETECT": "0",
        "NO_COLOR": "1",
        **env,
    }
    proc = subprocess.run(
        ["sh", str(ROOT / "install.sh")], env=full, capture_output=True, text=True, timeout=120
    )
    log = (tmp_path / "log").read_text() if (tmp_path / "log").exists() else ""
    return proc, log


def test_installer_with_uv(tmp_path):
    proc, log = run_installer(tmp_path, "uv")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "uv tool install --force --python python3" in log
    assert f"argus-harness[tui] @ file://{ROOT}" in log
    assert (tmp_path / "home/.config/argus/config.toml").read_text() == DEFAULT_CONFIG
    out = proc.stdout
    assert "==> installing argus with uv" in out
    assert "==> checking this machine" in out and f"argus {__version__} · Python" in out
    assert f"{tmp_path / 'bin'} is not on your PATH" in out
    assert "done." in out


def test_installer_with_pipx_keeps_an_existing_config(tmp_path):
    cfg = tmp_path / "home/.config/argus/config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("# mine\n")
    proc, log = run_installer(tmp_path, "pipx", ARGUS_NO_UV="1", PIPX_BIN_DIR=str(tmp_path / "bin"))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert log.startswith("pipx install --force --python python3")
    assert "kept" in proc.stdout and cfg.read_text() == "# mine\n"


def test_installer_reports_a_bad_source(tmp_path):
    proc, _ = run_installer(tmp_path, "uv", ARGUS_SOURCE=str(tmp_path / "missing"))
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


@pytest.mark.skipif(
    not (os.environ.get("ARGUS_TEST_INSTALL") and shutil.which("uv")),
    reason="set ARGUS_TEST_INSTALL=1 to install for real with uv (needs the package index)",
)
def test_installer_for_real(tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "UV_TOOL_DIR": str(tmp_path / "uv-tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
        "ARGUS_SOURCE": str(ROOT),
        "ARGUS_LOCAL_SERVERS": "",
        "NO_COLOR": "1",
    }
    for key in ("UV_CACHE_DIR", "HTTPS_PROXY", "SSL_CERT_FILE", "UV_INDEX_URL"):
        if key in os.environ:
            env[key] = os.environ[key]
    proc = subprocess.run(
        ["sh", str(ROOT / "install.sh")], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    argus = tmp_path / "bin/argus"
    version = subprocess.run([str(argus), "--version"], capture_output=True, text=True, env=env)
    assert version.stdout.strip() == f"argus {__version__}"
