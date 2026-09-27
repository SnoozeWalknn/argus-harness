"""write, ls, fetch, web_search and background jobs."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from argus.approval import FixedApprover, Gate
from argus.config import ToolsConfig
from argus.executors import LocalExecutor
from argus.mock import call, final
from argus.sandbox import (
    WORKSPACE_WRITE,
    Bwrap,
    Landlock,
    NoSandbox,
    Spec,
    bwrap_works,
    landlock_works,
)
from argus.ssh import SSHExecutor
from argus.tools import ToolContext, ToolError, build_tools
from argus.tools.jobs import Jobs
from argus.tools.web import html_to_text

SHIM = [sys.executable, str(Path(__file__).parent / "fake_ssh.py")]

PAGE = """<!doctype html><html><head><title>Calc docs</title><style>p{}</style>
<script>alert(1)</script></head><body><nav><a href="/">home</a></nav>
<h1>calc</h1><p>Adds &amp; multiplies <b>numbers</b>. See <a href="/api">the API</a>.</p>
<ul><li>add(a, b)</li><li>mul(a, b)</li></ul>
<pre><code>def add(a, b):
    return a + b
</code></pre><p>Use <code>add</code> first.</p></body></html>"""


class Site:
    """A local web server: pages by path, plus search API stand-ins."""

    def __init__(self):
        routes = {
            "/docs": (200, "text/html; charset=utf-8", PAGE.encode()),
            "/data.json": (200, "application/json", b'{"b": 1, "a": [1, 2]}'),
            "/long.txt": (200, "text/plain", ("0123456789" * 5000).encode()),
            "/img.png": (200, "image/png", b"\x89PNG\r\n\x1a\n" + bytes(100)),
        }
        self.requests: list[tuple[str, dict]] = []
        site = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urlsplit(self.path)
                site.requests.append((u.path, {**parse_qs(u.query), **dict(self.headers)}))
                if u.path == "/res/v1/web/search":
                    q = parse_qs(u.query)["q"][0]
                    body = {"web": {"results": [
                        {"title": f"Result for {q}", "url": "https://example.com/a", "description": "the <strong>first</strong> hit"},
                        {"title": "Second", "url": "https://example.com/b", "description": "another"},
                    ]}}  # fmt: skip
                    return self.send(200, "application/json", json.dumps(body).encode())
                if u.path == "/search":
                    body = {
                        "results": [
                            {
                                "title": "Searx hit",
                                "url": "https://s.example/x",
                                "content": "snippet",
                            }
                        ]
                    }
                    return self.send(200, "application/json", json.dumps(body).encode())
                status, ctype, body = routes.get(u.path, (404, "text/html", b"<h1>Not found</h1>"))
                self.send(status, ctype, body)

            def send(self, status, ctype, body):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def site():
    s = Site()
    yield s
    s.stop()


def ctx_for(workspace, executor=None, **cfg) -> tuple[ToolContext, dict]:
    tc = ToolsConfig(**cfg)
    ex = executor or LocalExecutor(str(workspace))
    return ToolContext(ex, tc, jobs=Jobs()), build_tools(tc)


# -- write ----------------------------------------------------------------------------------------


def test_write_creates_and_replaces_after_a_read(workspace):
    ctx, tools = ctx_for(workspace)
    r = tools["write"].run(ctx, {"path": "pkg/new.py", "content": "x = 1\ny = 2\n"})
    assert r.text == "Created pkg/new.py (2 lines)." and r.meta["created"]
    assert (workspace / "pkg/new.py").read_text() == "x = 1\ny = 2\n"
    with pytest.raises(ToolError, match="read calc.py before replacing it"):
        tools["write"].run(ctx, {"path": "calc.py", "content": "pass\n"})
    tools["read"].run(ctx, {"path": "calc.py"})
    (workspace / "calc.py").write_text("changed behind the model's back\n")
    with pytest.raises(ToolError, match="changed since you read it"):
        tools["write"].run(ctx, {"path": "calc.py", "content": "pass\n"})
    tools["read"].run(ctx, {"path": "calc.py"})
    r = tools["write"].run(ctx, {"path": "calc.py", "content": "def add(a, b:\n"})
    assert r.text.startswith("Replaced calc.py (1 lines).") and "syntax error" in r.text
    with pytest.raises(ToolError, match="outside the workspace"):
        tools["write"].run(ctx, {"path": "/etc/argus-test", "content": "x"})


def test_write_in_a_run_passes_the_fs_checks(make_agent, workspace):
    steps = [
        call("write", path="notes/todo.md", content="# todo\n- fix add\n"),
        final("wrote it"),
    ]
    agent, _ = make_agent(steps)
    result = agent.run("write a todo file")
    assert result.status == "completed" and result.failures == []
    assert (workspace / "notes/todo.md").exists()


def test_plan_mode_has_no_write(make_agent):
    agent, server = make_agent([final("plan")], overrides=['agent.mode="plan"'])
    agent.run("x")
    names = {t["function"]["name"] for t in server.requests[0]["tools"]}
    assert "write" not in names and "edit" not in names and "ls" in names


# -- ls ------------------------------------------------------------------------------------------------


def test_ls_tree_with_depth(workspace):
    (workspace / "pkg/deep/deeper").mkdir(parents=True)
    (workspace / "pkg/deep/deeper/x.py").write_text("")
    (workspace / "pkg/deep/y.py").write_text("")
    ctx, tools = ctx_for(workspace)
    out = tools["ls"].run(ctx, {}).text.splitlines()
    assert out[0] == "./  (7 files)"
    assert "  pkg/" in out and "    deep/  (2 files)" in out and "  calc.py" in out
    assert out.index("  pkg/") < out.index("  README.md")  # directories first
    deep = tools["ls"].run(ctx, {"path": "pkg", "depth": 5}).text
    assert deep.startswith("pkg/  (4 files)\n  deep/\n    deeper/\n      x.py\n    y.py")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools["ls"].run(ctx, {"path": "/etc"})


# -- fetch -----------------------------------------------------------------------------------------------


def test_html_to_text():
    title, text = html_to_text(PAGE, "https://calc.example/docs")
    assert title == "Calc docs"
    assert "alert" not in text and "p{}" not in text
    assert "# calc" in text
    assert "Adds & multiplies numbers. See the API (https://calc.example/api)." in text
    assert "- add(a, b)\n- mul(a, b)" in text
    assert "```\ndef add(a, b):\n    return a + b\n" in text
    assert "Use `add` first." in text


def test_fetch_pages(workspace, site):
    ctx, tools = ctx_for(workspace, fetch_max_chars=1000)
    fetch = tools["fetch"]
    r = fetch.run(ctx, {"url": f"{site.url}/docs"})
    assert r.ok and r.text.startswith(f"{site.url}/docs  [200 text/html]  Calc docs")
    assert "# calc" in r.text
    j = fetch.run(ctx, {"url": f"{site.url}/data.json"})
    assert '"a": [\n    1,' in j.text
    long = fetch.run(ctx, {"url": f"{site.url}/long.txt"})
    assert long.truncated and "[49000 more characters; fetch again with start=1000]" in long.text
    nxt = fetch.run(ctx, {"url": f"{site.url}/long.txt", "start": 49500})
    assert not nxt.truncated and nxt.text.endswith("0123456789")
    img = fetch.run(ctx, {"url": f"{site.url}/img.png"})
    assert "108 bytes of image/png content, not shown" in img.text
    missing = fetch.run(ctx, {"url": f"{site.url}/nope"})
    assert not missing.ok and missing.error == "HTTP 404" and "Not found" in missing.text
    with pytest.raises(ToolError, match="only http and https"):
        fetch.run(ctx, {"url": "file:///etc/passwd"})
    with pytest.raises(ToolError, match="could not fetch"):
        fetch.run(ctx, {"url": "http://127.0.0.1:9/"})


# -- web search ---------------------------------------------------------------------------------------------


def test_web_search_brave_and_searxng(workspace, site, monkeypatch):
    ctx, tools = ctx_for(workspace)
    assert "web_search" not in tools  # nothing configured
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    ctx, tools = ctx_for(workspace, search_url=f"{site.url}/res/v1/web/search")
    r = tools["web_search"].run(ctx, {"query": "textual pilot", "count": 2})
    assert r.text.startswith(
        "1. Result for textual pilot\n   https://example.com/a\n   the first hit"
    )
    path, seen = site.requests[-1]
    assert seen["X-Subscription-Token"] == "brave-test-key" and seen["count"] == ["2"]
    monkeypatch.delenv("BRAVE_API_KEY")
    monkeypatch.setenv("SEARXNG_URL", site.url)
    ctx, tools = ctx_for(workspace)
    r = tools["web_search"].run(ctx, {"query": "x"})
    assert r.text == "1. Searx hit\n   https://s.example/x\n   snippet"
    assert site.requests[-1][1]["format"] == ["json"]
    ctx, tools = ctx_for(workspace, search="off")
    assert "web_search" not in tools


# -- the approval gate for network tools ----------------------------------------------------------------------


def test_network_tools_follow_the_gate():
    off = Gate("auto", NoSandbox(), FixedApprover("yes"), network=False)
    d = off.decide("fetch", {"url": "https://x.dev"}, False, True)
    assert not d.allow and "sandbox.network" in d.reason
    assert (
        Gate("read-only", NoSandbox(), FixedApprover("no"))
        .decide("fetch", {"url": "u"}, False, True)
        .allow
    )
    asked: list[str] = []

    class Rec(FixedApprover):
        def ask(self, req):
            asked.append(req.summary)
            return "always"

    ask = Gate("ask", NoSandbox(), Rec("yes"))
    assert ask.decide("fetch", {"url": "https://docs.python.org/3/"}, False, True).allow
    assert ask.decide("fetch", {"url": "https://docs.python.org/3/library"}, False, True).allow
    assert ask.decide("fetch", {"url": "https://pypi.org/"}, False, True).allow
    assert asked == ["fetch: https://docs.python.org/3/", "fetch: https://pypi.org/"]  # per host
    assert ask.decide("web_search", {"query": "q"}, False, True).allow


def test_fetch_denied_in_a_run_without_network(make_agent, site):
    steps = [call("fetch", url=f"{site.url}/docs"), final("no network")]
    agent, _ = make_agent(steps, overrides=["sandbox.network=false"])
    result = agent.run("read the docs")
    tc = agent.store.tool_calls(result.run_id)[0]
    assert not tc["ok"] and "network access is off" in tc["result"]
    assert site.requests == []


# -- background jobs ----------------------------------------------------------------------------------------------


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.05)


def test_background_job_lifecycle(workspace):
    ctx, tools = ctx_for(workspace, job_start_wait=2.0)
    bash, job = tools["bash"], tools["job"]
    r = bash.run(
        ctx,
        {
            "cmd": "echo ready; sleep 1.5; for i in 1 2 3; do sleep 0.2; echo tick $i; done",
            "background": True,
        },
    )
    assert r.text.startswith("started job 1: echo ready")
    assert "ready" in r.text and "[job 1 running (pid " in r.text
    out = job.run(ctx, {"action": "output", "id": 1, "wait": 10})
    assert "tick 3" in out.text and "ready" not in out.text  # only new output
    assert out.text.endswith("[job 1 exited 0]")
    assert job.run(ctx, {"action": "output", "id": 1}).text == "(no new output)\n[job 1 exited 0]"
    quick = bash.run(ctx, {"cmd": "echo fast; exit 3", "background": True})
    assert "fast" in quick.text and quick.text.endswith("[job 2 exited 3]")
    server = bash.run(ctx, {"cmd": "echo serving; sleep 60", "background": True})
    assert "[job 3 running" in server.text
    listing = job.run(ctx, {"action": "list"}).text
    assert "1  exited 0" in listing and "3  running" in listing
    killed = job.run(ctx, {"action": "kill", "id": 3})
    assert killed.text.endswith("[job 3 killed]")
    with pytest.raises(ToolError, match="no job 9"):
        job.run(ctx, {"action": "output", "id": 9})


def test_jobs_limit_and_close(workspace):
    ex = LocalExecutor(str(workspace))
    jobs = Jobs(limit=2)
    a = jobs.start(ex, "sleep 60", None)
    jobs.start(ex, "sleep 60", None)
    with pytest.raises(ToolError, match="2 background jobs are already running"):
        jobs.start(ex, "sleep 60", None)
    jobs.close()
    wait_for(lambda: a.handle.poll() is not None)
    assert all(j.status() == "killed" for j in jobs.jobs.values())


SANDBOXES = [
    pytest.param(Bwrap, marks=pytest.mark.skipif(not bwrap_works(), reason="bwrap unavailable")),
    pytest.param(Landlock, marks=pytest.mark.skipif(not landlock_works(), reason="no landlock")),
]


@pytest.mark.parametrize("backend", SANDBOXES)
def test_background_jobs_stay_sandboxed(workspace, backend):
    import tempfile

    outside = tempfile.mkdtemp(prefix="argus-outside-", dir="/var/tmp")
    ex = LocalExecutor(str(workspace))
    ex.sandbox_backend = backend()
    ctx = ToolContext(ex, ToolsConfig(job_start_wait=5.0), jobs=Jobs())
    ctx.sandbox = Spec(WORKSPACE_WRITE, str(workspace))
    tools = build_tools(ToolsConfig())
    r = tools["bash"].run(
        ctx,
        {
            "cmd": f"sleep 0.2; touch {outside}/escaped; echo in > inside.txt; echo done",
            "background": True,
        },
    )
    assert "done" in r.text and r.meta["sandbox"] == WORKSPACE_WRITE
    assert (workspace / "inside.txt").exists()
    assert not Path(outside, "escaped").exists()
    ctx.jobs.close()


def test_background_jobs_over_ssh(workspace):
    ex = SSHExecutor("vm", str(workspace), ssh_command=SHIM)
    ctx, tools = ctx_for(workspace, executor=ex, job_start_wait=2.0)
    try:
        r = tools["bash"].run(ctx, {"cmd": "echo remote; sleep 2; echo later", "background": True})
        assert "remote" in r.text
        out = tools["job"].run(ctx, {"action": "output", "id": 1, "wait": 10})
        assert "later" in out.text and out.text.endswith("[job 1 exited 0]")
        tools["bash"].run(ctx, {"cmd": "sleep 60", "background": True})
        assert tools["job"].run(ctx, {"action": "kill", "id": 2}).text.endswith("[job 2 killed]")
    finally:
        ctx.jobs.close()
        ex.close()


def test_a_run_starts_a_server_and_checks_it(make_agent, workspace):
    port_file = workspace / "port.txt"
    serve = (
        f'{sys.executable} -c "import http.server as h, socketserver as s; '
        "srv = s.TCPServer(('127.0.0.1', 0), h.SimpleHTTPRequestHandler); "
        "open('port.txt', 'w').write(str(srv.server_address[1])); print('listening', flush=True); "
        'srv.serve_forever()"'
    )

    def check(req, server):  # the model fetches the page the server serves
        port = port_file.read_text().strip()
        return call("fetch", url=f"http://127.0.0.1:{port}/README.md")

    steps = [call("bash", cmd=serve, background=True), check, final("server is up")]
    agent, _ = make_agent(steps)
    result = agent.run("start a server and check it")
    assert result.status == "completed", result.failures
    started, fetched = agent.store.tool_calls(result.run_id)[:2]
    assert "listening" in started["result"] and "[job 1 running" in started["result"]
    assert "A tiny calculator." in fetched["result"]
    job = agent.jobs.jobs[1]
    assert job.status() == "running"
    agent.close()  # argus run stops its jobs when it is done
    wait_for(lambda: job.handle.poll() is not None)
