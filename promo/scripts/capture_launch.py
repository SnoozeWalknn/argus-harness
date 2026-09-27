"""Capture everything the 90-second launch film shows, from real argus runs.

Nothing on screen is drawn by hand. The TUI states are SVG screenshots of the real
Textual app, driven with Textual's Pilot against argus's own mock server (no real
model is involved); the terminal text is the real output of argus commands. Writes:

* ``promo/public/launch/*.svg``: TUI screenshots (fonts rewritten to JetBrains Mono)
* ``promo/src/launch/data.json``: numbers and terminal lines

Two cosmetic changes, both labelled in the film: cloud providers show their real API
host instead of the mock's address, and the mock listens on the local servers' usual
ports (llama-server :8080, Ollama :11434) so URLs look like a real machine's.

Run from the repository root with the project venv:

    .venv/bin/python promo/scripts/capture_launch.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_SVG = ROOT / "promo/public/launch"
OUT_DATA = ROOT / "promo/src/launch/data.json"
WORK = Path(tempfile.mkdtemp(prefix="argus-launch-"))

os.environ.update(
    HOME=str(WORK / "home"),
    XDG_STATE_HOME=str(WORK / "state"),
    ARGUS_LSP_AUTODETECT="0",
    ARGUS_LOCAL_SERVERS="",
    ARGUS_CONFIG_DIR=str(WORK / "home/.config/argus"),
    PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",  # pytest for the scenes
)
for k in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
):
    os.environ.pop(k, None)
(WORK / "home").mkdir(parents=True)
sys.path.insert(0, str(ROOT))

from argus.cli import main as argus_main  # noqa: E402
from argus.cli import model_overrides  # noqa: E402
from argus.config import load_config  # noqa: E402
from argus.mock import MockServer, Script, call, final, raw_call  # noqa: E402
from argus.providers.base import Provider  # noqa: E402
from argus.tui.app import ArgusApp  # noqa: E402

CLOUD_HOSTS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openrouter": "https://openrouter.ai/api/v1",
}
_describe = Provider.describe
Provider.describe = lambda self: (  # type: ignore[method-assign]
    f"{self.kind} {CLOUD_HOSTS[self.kind]}" if self.kind in CLOUD_HOSTS else _describe(self)
)

SIZE = (140, 40)

CART = """from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    qty: int = 1


def total(items: list[Item]) -> float:
    return sum(i.price for i in items)


def apply_discount(amount: float, pct: float) -> float:
    return round(amount * (1 - pct / 100), 2)
"""
TEST_CART = """from cart import Item, total


def test_total_counts_quantity():
    assert total([Item("tea", 4.0, qty=3)]) == 12.0
"""


def workspace(name: str) -> Path:
    ws = WORK / name
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "cart.py").write_text(CART)
    (ws / "test_cart.py").write_text(TEST_CART)
    (ws / "README.md").write_text("# shop\n\nA tiny checkout.\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "init"],
        cwd=ws,
        check=True,
    )
    return ws


def svg_fix(svg: str) -> str:
    """Use the film's monospace font and drop Textual's web-font references."""
    svg = re.sub(r"@font-face\s*\{[^}]*\}", "", svg)
    svg = svg.replace(
        "font-family: Fira Code, monospace", 'font-family: "JetBrains Mono", monospace'
    )
    svg = svg.replace("font-family: arial", 'font-family: "Inter", sans-serif')
    return svg


def save(app: ArgusApp, name: str) -> None:
    OUT_SVG.mkdir(parents=True, exist_ok=True)
    tmp = WORK / f"{name}.svg"
    app.save_screenshot(str(tmp))
    (OUT_SVG / f"{name}.svg").write_text(svg_fix(tmp.read_text()))
    print("  saved", name)


async def until(pilot, cond, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise TimeoutError("TUI did not get there")
        await pilot.pause(0.05)


async def send(app, pilot, text: str) -> None:
    done = app.runs_done
    app.query_one("#prompt").value = text
    await pilot.press("enter")
    await until(pilot, lambda: app.runs_done > done)
    await pilot.pause(0.3)
    app.drain()
    await pilot.pause(0.2)


def maker(server: MockServer, ws: Path, *extra: str, theme: str = "tokyo-night"):
    db = WORK / "argus.db"

    def make(spec):
        overrides = (
            model_overrides(spec)
            if spec
            else ['model.provider="llama_server"', 'model.model="qwen3-coder-30b"']
        )
        return load_config(
            None,
            overrides
            + [
                f'model.base_url="{server.url}"',
                f'executor.workdir="{ws}"',
                f'log.db="{db}"',
                "model.retries=0",
                "lsp.enabled=false",
                "tools.todo='on'",
                f"tui.theme='{theme}'",
                *extra,
            ],
        )

    return make


THINK = (
    "test_total_counts_quantity expects 3 teas at 4.0 to cost 12.0, but total() sums "
    "prices only, so quantity is ignored. I'll read cart.py to confirm, fix total(), then run the test."
)
TODO = [
    {"content": "read cart.py", "status": "done"},
    {"content": "count quantity in total()", "status": "in_progress"},
    {"content": "run the tests", "status": "pending"},
]


def hero_steps():
    return [
        {"reasoning": THINK, **call("read", path="cart.py")},
        {"reasoning": "Track the plan.", **call("todo", items=TODO)},
        {
            "reasoning": "total() sums i.price; it should multiply by i.qty.",
            **call(
                "edit",
                path="cart.py",
                old="sum(i.price for i in items)",
                new="sum(i.price * i.qty for i in items)",
            ),
        },
        {"reasoning": "Run the test.", **call("bash", cmd="pytest -q test_cart.py")},
        final(
            "Fixed **total()** in `cart.py`: it summed prices and ignored `qty`.\n\n"
            "- `test_total_counts_quantity` passes\n"
            "- `apply_discount` looked fine; I left it alone"
        ),
    ]


async def scene_hero() -> dict:
    """The main session: thinking, token box, tools, then /model opus and a follow-up."""
    ws = workspace("hero")
    server = MockServer(
        Script(
            hero_steps()
            + [
                {
                    "reasoning": "The user wants a guard for negative quantities. Add a check in Item.__post_init__.",
                    **call(
                        "edit",
                        path="cart.py",
                        old="    qty: int = 1\n",
                        new='    qty: int = 1\n\n    def __post_init__(self):\n        if self.qty < 0:\n            raise ValueError("qty must be >= 0")\n',
                    ),
                },
                final(
                    "Added a guard: `Item(qty=-1)` now raises `ValueError`. The session history carried over from qwen3-coder."
                ),
            ]
        ),
        chunk_delay=0.035,
        port=8080,
        model="qwen3-coder-30b",
    ).start()
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-demo"
    os.environ["GEMINI_API_KEY"] = "AIza-demo"
    os.environ["OPENROUTER_API_KEY"] = "sk-or-demo"
    ollama = MockServer(Script([]), port=11434, flavor="ollama", model="qwen3-coder:30b").start()
    os.environ["ARGUS_LOCAL_SERVERS"] = f"ollama={ollama.url}"
    app = ArgusApp(maker(server, ws, "agent.protocol='native'"))
    info = {}
    try:
        async with app.run_test(size=SIZE) as pilot:
            app.query_one("#prompt").value = "the checkout test fails — fix it"
            await pilot.press("enter")
            await pilot.pause(2.4)
            app.drain()
            app.refresh_meter()
            await pilot.pause(0.1)
            save(app, "hero-thinking")
            await until(pilot, lambda: app.runs_done == 1)
            await pilot.pause(0.3)
            app.drain()
            await pilot.pause(0.2)
            save(app, "hero-done")
            info["hero_tokens"] = [app.totals["prompt"], app.totals["gen"]]
            # the model picker
            await pilot.press("ctrl+o")
            await until(pilot, lambda: len(app.screen.query("#picker-input")) == 1)
            await pilot.pause(0.3)
            save(app, "picker")
            await pilot.press(*"opus")
            await pilot.pause(0.2)
            save(app, "picker-filtered")
            await pilot.press("escape")
            await pilot.pause(0.2)
            # /model opus and a follow-up on Claude, same session
            app.query_one("#prompt").value = "/model opus"
            await pilot.press("enter")
            await pilot.pause(0.3)
            save(app, "switched")
            await send(app, pilot, "also reject negative quantities")
            save(app, "switched-done")
            # help
            app.query_one("#prompt").value = "/help"
            await pilot.press("enter")
            await pilot.pause(0.3)
            save(app, "help")
            await pilot.press("escape")
            await pilot.pause(0.1)
            # themes, on the finished session
            for theme in (
                "matte-black",
                "kanagawa",
                "everforest",
                "osaka-jade",
                "gruvbox",
                "catppuccin-mocha",
                "rose-pine",
                "nord",
            ):
                app.theme = theme
                await pilot.pause(0.25)
                save(app, f"theme-{theme}")
    finally:
        server.stop()
        ollama.stop()
        os.environ["ARGUS_LOCAL_SERVERS"] = ""
    return info


async def scene_approval() -> None:
    ws = workspace("approval")
    steps = [
        {"reasoning": "Dependencies first.", **call("bash", cmd="npm install --save-dev vitest")},
        final("Installed vitest."),
    ]
    server = MockServer(Script(steps), chunk_delay=0.01, port=8080).start()
    app = ArgusApp(maker(server, ws, 'agent.approval="ask"'))
    try:
        async with app.run_test(size=SIZE) as pilot:
            app.query_one("#prompt").value = "add vitest"
            await pilot.press("enter")
            await until(pilot, lambda: len(app.screen.query("#approval-summary")) == 1)
            await pilot.pause(0.3)
            save(app, "approval")
            await pilot.press("n")
            await until(pilot, lambda: app.runs_done == 1)
    finally:
        server.stop()


async def scene_plan() -> None:
    ws = workspace("plan")
    plan = (
        "**Plan: discount codes**\n\n"
        "1. `cart.py`: add `Code(name, pct, expires)` and `apply_code(items, code)`\n"
        "2. reuse `apply_discount()`; reject expired codes with `ValueError`\n"
        "3. `test_cart.py`: valid, expired and unknown codes\n"
        "4. verify: `pytest -q`\n\n"
        "No files changed. Switch to build mode (tab) to carry it out."
    )
    steps = [
        {
            "reasoning": "Plan mode: read-only. Look at the cart first.",
            **call("read", path="cart.py"),
        },
        {"reasoning": "Where is apply_discount used?", **call("grep", pattern="apply_discount")},
        final(plan),
    ]
    server = MockServer(Script(steps), chunk_delay=0.005, port=8080).start()
    app = ArgusApp(maker(server, ws, 'agent.mode="plan"'))
    try:
        async with app.run_test(size=SIZE) as pilot:
            await send(app, pilot, "how would we add discount codes?")
            save(app, "plan")
    finally:
        server.stop()


async def scene_subagent_jobs() -> None:
    ws = workspace("jobs")
    serve = "python3 -m http.server 5173"
    steps = [
        {
            "reasoning": "Delegate the search to a subagent with a fresh context.",
            **call("task", agent="explore", prompt="Find every place prices are summed."),
        },
        call("grep", pattern="sum\\("),
        final("Only total() in cart.py:12 sums prices."),
        {
            "reasoning": "Start the dev server in the background, then check it.",
            **call("bash", cmd=serve, background=True),
        },
        call("fetch", url="http://localhost:5173/README.md"),
        final("The dev server is up on :5173 (job 1) and serves the README."),
    ]
    server = MockServer(Script(steps), chunk_delay=0.005, port=8080).start()
    app = ArgusApp(maker(server, ws, 'agent.subagents="on"', "tools.job_start_wait=1.5"))
    try:
        async with app.run_test(size=SIZE) as pilot:
            await send(app, pilot, "where are prices summed? then start the dev server")
            app.refresh_jobs()
            await pilot.pause(0.2)
            save(app, "subagent-jobs")
    finally:
        server.stop()


async def scene_lsp() -> None:
    ws = workspace("lsp")
    pylsp = Path(sys.executable).parent / "pylsp"  # the real python-lsp-server, with pyflakes
    server_cmd = (
        [str(pylsp)] if pylsp.exists() else [sys.executable, str(ROOT / "tests/fake_lsp.py")]
    )
    fake_lsp = json.dumps(server_cmd)
    steps = [
        call("read", path="cart.py"),
        {
            "reasoning": "Rename the helper.",
            **call(
                "edit",
                path="cart.py",
                old="return round(amount * (1 - pct / 100), 2)",
                new="return round(amount * (1 - percent / 100), 2)",
            ),
        },
        {
            "reasoning": "The language server says `percent` is undefined; the parameter is still pct.",
            **call("edit", path="cart.py", old="(1 - percent / 100)", new="(1 - pct / 100)"),
        },
        final("Reverted the rename: the parameter is still `pct`, so `percent` was undefined."),
    ]
    server = MockServer(Script(steps), chunk_delay=0.005, port=8080).start()
    app = ArgusApp(
        maker(
            server,
            ws,
            "lsp.enabled=true",
            "lsp.wait_ms=20000",
            f"lsp.servers={{python = {fake_lsp}}}",
        )
    )
    try:
        async with app.run_test(size=SIZE) as pilot:
            await send(app, pilot, "rename pct to percent in apply_discount")
            save(app, "lsp")
    finally:
        server.stop()


async def scene_refusal() -> None:
    ws = workspace("refusal")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-demo"
    server = MockServer(
        Script(
            [{"content": "", "refusal": {"category": "cyber"}}, final("Here is a safe version.")]
        ),
        port=8080,
    ).start()
    make = maker(server, ws)

    def claude(spec):
        return make(spec or "sonnet")

    app = ArgusApp(claude)
    try:
        async with app.run_test(size=SIZE) as pilot:
            await send(app, pilot, "write an exploit for this CVE")
            save(app, "refusal")
    finally:
        server.stop()


# -- terminal output ---------------------------------------------------------------------------------


def capture(argv: list[str]) -> str:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            argus_main(argv)
        except SystemExit:
            pass
    return (out.getvalue() + err.getvalue()).replace(str(WORK / "home"), "~")


def doctor_lines() -> list[str]:
    server = MockServer(
        Script([]), port=8080, model="Qwen3-Coder-30B-A3B-Q4_K_M", n_ctx=65536
    ).start()
    os.environ["ARGUS_LOCAL_SERVERS"] = f"llama_server={server.url}"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-demo"
    try:
        capture(["init"])
        text = capture(["doctor"])
    finally:
        server.stop()
        os.environ["ARGUS_LOCAL_SERVERS"] = ""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"Linux \S+", "Linux", line)
        if line.startswith(
            ("argus ", "config", "sandbox", "git", "rg", "bwrap", "tui", "api keys", "model server")
        ):
            lines.append(line)
    return lines


def tune_lines() -> dict:

    def policy(req, server):
        done = any(
            m.get("role") == "tool" or "<tool_response>" in str(m.get("content"))
            for m in req["messages"]
        )
        native = bool(req.get("tools"))
        if native:
            return raw_call("edit", '{"path": "calc.py", "old": ')
        if done:
            return final("Fixed.")
        return call("edit", path="calc.py", old="return a - b", new="return a + b")

    suite = WORK / "tune/suite.toml"
    (WORK / "tune/fx").mkdir(parents=True, exist_ok=True)
    (WORK / "tune/fx/calc.py").write_text("def add(a, b):\n    return a - b\n")
    suite.write_text(
        'name = "tune"\n[[task]]\nid = "fix-add"\nprompt = "fix add"\nworkspace = "fx"\n'
        "check = \"grep -q 'a + b' calc.py\"\n"
        'overrides = ["tools.require_read_before_edit=false"]\n'
    )
    server = MockServer(Script.always(policy)).start()
    try:
        text = capture(
            [
                "tune",
                "llama/qwen3-coder-30b",
                "--suite",
                str(suite),
                "-n",
                "3",
                "-q",
                "--dry-run",
                "-o",
                f'model.base_url="{server.url}"',
                "-o",
                f'log.db="{WORK / "tune.db"}"',
            ]
        )
    finally:
        server.stop()
    lines = text.splitlines()
    table = [ln for ln in lines if re.match(r"^(variant|native|json_schema|grammar)\s", ln)]
    head = next((i for i, ln in enumerate(lines) if "lower95" in ln), len(lines))
    ranking = lines[head : head + 4]
    best = next((ln for ln in lines if ln.startswith("best:")), "")
    return {"table": table[:4], "ranking": ranking[:4], "best": best}


def overhead() -> dict:
    from argus.agent import Agent
    from argus.tokens import measure

    server = MockServer(Script([])).start()
    out = {}
    try:
        ws = workspace("overhead")
        for proto in ("native", "json_schema", "grammar"):
            cfg = load_config(
                None,
                [
                    f'model.base_url="{server.url}"',
                    f'executor.workdir="{ws}"',
                    f'agent.protocol="{proto}"',
                    "lsp.enabled=false",
                    f'log.db="{WORK / "o.db"}"',
                ],
            )
            a = Agent(cfg)
            out[proto] = measure(a).total
            a.close()
    finally:
        server.stop()
    return out


def install_lines() -> list[str]:
    """The installer's own output, with stand-ins for uv (no network)."""
    tools = WORK / "fakebin"
    tools.mkdir(exist_ok=True)
    fake = (
        '#!/bin/sh\nif [ "$1 $2" = "tool dir" ]; then echo "$HOME/.local/bin"; exit 0; fi\n'
        'mkdir -p "$HOME/.local/bin"\n'
        f'printf \'#!/bin/sh\\nPYTHONPATH="{ROOT}" exec "{sys.executable}" -m argus "$@"\\n\' > "$HOME/.local/bin/argus"\n'
        'chmod +x "$HOME/.local/bin/argus"\n'
    )
    (tools / "uv").write_text(fake)
    (tools / "uv").chmod(0o755)
    home = WORK / "installhome"
    home.mkdir(exist_ok=True)
    env = {
        "PATH": f"{tools}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(home),
        "ARGUS_SOURCE": str(ROOT),
        "ARGUS_LOCAL_SERVERS": "",
        "ARGUS_LSP_AUTODETECT": "0",
        "NO_COLOR": "1",
    }
    r = subprocess.run(
        ["sh", str(ROOT / "install.sh")], env=env, capture_output=True, text=True, timeout=120
    )
    text = r.stdout.replace(str(home), "~").replace(
        f"file://{ROOT}", "git+https://github.com/SnoozeWalknn/argus-harness@main"
    )
    return [re.sub(r"Linux \S+", "Linux", ln) for ln in text.splitlines() if ln.strip()]


def test_count() -> int:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return sum(int(m.group(1)) for m in re.finditer(r"^tests/\S+\.py: (\d+)$", r.stdout, re.M))


TEXT_RUN = re.compile(
    r'<text class="[^"]*" x="([0-9.]+)" y="([0-9.]+)" textLength="([0-9.]+)"[^>]*>([^<]*)</text>'
)
ANCHORS = {
    "hero-thinking": {"thinking": "∴ thinking", "meter": "=tokens", "rate": "tok ·"},
    "hero-done": {
        "todo": "=todo", "changed": "=changed", "final": "Fixed", "tests": "passed in",
        "meter": "=tokens", "spark": "tok/s", "status": "qwen3-coder-30b",
    },
    "picker": {"title": "switch model", "opus": "opus", "needs": "needs OPENAI_API_KEY", "recent": "recent"},
    "picker-filtered": {"opus": "opus"},
    "switched-done": {
        "status": "claude-opus-5-5", "continues": "the session continues",
        "edited": "Edited cart.py (lines 8-12)", "user": "also reject",
    },
    "subagent-jobs": {
        "explore": "explore │ ▸ grep", "job": "started job 1", "fetch": "fetch(url=", "jobs": "1 ●",
    },
    "approval": {"cmd": "bash: npm install", "always": "always"},
    "plan": {"badge": "PLAN", "title": "Plan: discount codes"},
    "lsp": {"warn": "after this edit:", "error": "undefined name"},
    "refusal": {"declined": "the model declined"},
}  # fmt: skip


def anchors() -> dict:
    """Where named strings sit in each SVG (viewBox units): [x, y centre, width]."""
    import html as _html

    out = {}
    for name, wanted in ANCHORS.items():
        svg = (OUT_SVG / f"{name}.svg").read_text()
        vb = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
        # the text sits in the terminal group (translate(9, 41)); the window chrome's own
        # group comes first and is closed before it, so its offset does not apply
        term = re.search(r'<g transform="translate\(([0-9.]+),\s*([0-9.]+)\)" clip-path', svg)
        ox, oy = float(term.group(1)), float(term.group(2))
        runs = [
            (float(x) + ox, float(y) + oy, float(w), _html.unescape(t).replace("\xa0", " "))
            for x, y, w, t in TEXT_RUN.findall(svg)
        ]
        found = {"size": [float(vb.group(1)), float(vb.group(2))]}
        for key, needle in wanted.items():
            exact = needle.startswith("=")  # "=todo": a run that is exactly this text
            needle = needle.lstrip("=")
            for x, y, w, t in runs:
                i = (
                    t.find(needle)
                    if not exact
                    else (t.index(needle) if t.strip() == needle else -1)
                )
                if i >= 0:
                    cw = w / max(len(t), 1)
                    found[key] = [round(x + i * cw, 1), round(y - 7, 1), round(len(needle) * cw, 1)]
                    break
            else:
                print(f"  anchor not found: {name}:{key} ({needle!r})")
        out[name] = found
    return out


async def all_scenes() -> dict:
    info = await scene_hero()
    await scene_approval()
    await scene_plan()
    await scene_subagent_jobs()
    await scene_lsp()
    await scene_refusal()
    return info


def main() -> None:
    if "--anchors" in sys.argv:  # recompute anchors for the SVGs already captured
        data = json.loads(OUT_DATA.read_text())
        data["anchors"] = anchors()
        OUT_DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return
    print("TUI scenes")
    info = asyncio.run(all_scenes())
    print("terminal output")
    data = {
        **info,
        "overhead": overhead(),
        "doctor": doctor_lines(),
        "tune": tune_lines(),
        "install": install_lines(),
        "models": [ln for ln in capture(["models"]).splitlines() if ln.startswith("  ")][-10:],
        "tests": test_count(),
        "anchors": anchors(),
    }
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: v for k, v in data.items() if k != "install"}, indent=2, ensure_ascii=False)[
            :3000
        ]
    )
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
