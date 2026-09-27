"""Language-server diagnostics after edits (as in OpenCode).

A minimal LSP client over stdin/stdout: argus starts the right server for a
file's language on first use (through the executor, so it runs where the
workspace is, SSH included), opens or updates the edited file, waits briefly for
``textDocument/publishDiagnostics`` and hands the errors back so they can be
appended to the edit's result. The model sees a type error or a broken import
right after the edit that caused it instead of turns later.

Servers are looked up per language from ``lsp.servers`` or, by default, the
first of the known servers that is installed: pyright / basedpyright / pylsp for
Python, typescript-language-server, gopls, rust-analyzer, clangd.
"""

from __future__ import annotations

import json
import os
import posixpath
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urlsplit

LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {
        "exts": {".py": "python", ".pyi": "python"},
        "servers": [
            ["pyright-langserver", "--stdio"],
            ["basedpyright-langserver", "--stdio"],
            ["pylsp"],
        ],
    },
    "typescript": {
        "exts": {
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".mjs": "javascript",
            ".cjs": "javascript",
        },
        "servers": [["typescript-language-server", "--stdio"]],
    },
    "go": {"exts": {".go": "go"}, "servers": [["gopls"]]},
    "rust": {"exts": {".rs": "rust"}, "servers": [["rust-analyzer"]]},
    "c": {
        "exts": {".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cxx": "cpp"},
        "servers": [["clangd"]],
    },
}
SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def uri_for(path: str) -> str:
    return "file://" + quote(path)


def path_for(uri: str) -> str:
    return unquote(urlsplit(uri).path)


def language_of(path: str) -> tuple[str, str] | None:
    """(language, languageId) for a file, or None."""
    ext = posixpath.splitext(path)[1].lower()
    for lang, spec in LANGUAGES.items():
        if ext in spec["exts"]:
            return lang, spec["exts"][ext]
    return None


@dataclass
class Diagnostic:
    line: int  # 1-based
    col: int
    severity: str
    message: str
    source: str = ""

    def render(self, rel: str) -> str:
        src = f" ({self.source})" if self.source else ""
        return f"{rel}:{self.line}:{self.col} {self.severity}: {self.message}{src}"


class LSPError(RuntimeError):
    pass


class LSPClient:
    """One language server process."""

    def __init__(self, argv: list[str], cwd: str | None, root: str, timeout: float = 10.0):
        self.argv = argv
        self.root = root
        self.timeout = timeout
        self.proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._id = 0
        self._lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._diags: dict[str, tuple[int, list[dict[str, Any]]]] = {}  # uri -> (seq, items)
        self._cond = threading.Condition()
        self._seq = 0
        self.versions: dict[str, int] = {}
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()

    # -- transport ---------------------------------------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        body = json.dumps({"jsonrpc": "2.0", **msg}).encode()
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        with self._lock:
            try:
                self.proc.stdin.write(data)  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError) as e:
                self.alive = False
                raise LSPError(f"language server {self.argv[0]} is gone: {e}") from None

    def _read_message(self) -> dict[str, Any] | None:
        out = self.proc.stdout
        length = None
        while True:
            line = out.readline()  # type: ignore[union-attr]
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            name, _, value = line.decode("ascii", "replace").partition(":")
            if name.lower() == "content-length":
                length = int(value.strip())
        if length is None:
            return {}
        return json.loads(out.read(length))  # type: ignore[union-attr]

    def _reader(self) -> None:
        try:
            while True:
                msg = self._read_message()
                if msg is None:
                    break
                if not msg:
                    continue
                if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
                    q = self._pending.pop(msg["id"], None)
                    if q is not None:
                        q.put(msg)
                elif msg.get("method") == "textDocument/publishDiagnostics":
                    p = msg.get("params") or {}
                    with self._cond:
                        self._seq += 1
                        self._diags[p.get("uri", "")] = (self._seq, p.get("diagnostics") or [])
                        self._cond.notify_all()
                elif "id" in msg and "method" in msg:  # a request from the server
                    self._answer(msg)
        except (OSError, ValueError):
            pass
        finally:
            self.alive = False
            with self._cond:
                self._cond.notify_all()

    def _answer(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        if method == "workspace/configuration":
            items = (msg.get("params") or {}).get("items") or []
            result: Any = [None] * len(items)
        elif method == "workspace/workspaceFolders":
            result = [{"uri": uri_for(self.root), "name": posixpath.basename(self.root)}]
        else:  # registerCapability, workDoneProgress/create, ...: accept
            result = None
        try:
            self._send({"id": msg["id"], "result": result})
        except LSPError:
            pass

    def request(self, method: str, params: Any, timeout: float | None = None) -> Any:
        with self._lock:
            self._id += 1
            rid = self._id
        q: queue.Queue = queue.Queue()
        self._pending[rid] = q
        self._send({"id": rid, "method": method, "params": params})
        try:
            msg = q.get(timeout=timeout or self.timeout)
        except queue.Empty:
            self._pending.pop(rid, None)
            raise LSPError(f"{self.argv[0]}: no answer to {method}") from None
        if "error" in msg:
            raise LSPError(f"{self.argv[0]}: {method} failed: {msg['error']}")
        return msg.get("result")

    def notify(self, method: str, params: Any) -> None:
        self._send({"method": method, "params": params})

    # -- protocol ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        name = posixpath.basename(self.root) or "workspace"
        self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": uri_for(self.root),
                "rootPath": self.root,
                "workspaceFolders": [{"uri": uri_for(self.root), "name": name}],
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didSave": True, "dynamicRegistration": False},
                        "publishDiagnostics": {"relatedInformation": False},
                    },
                    "workspace": {"configuration": True, "workspaceFolders": True},
                    "window": {"workDoneProgress": True},
                },
                "clientInfo": {"name": "argus"},
            },
            timeout=max(self.timeout, 30),
        )
        self.notify("initialized", {})

    def update(self, path: str, text: str, language_id: str) -> int:
        """Open or change a document; returns the diagnostics sequence number before it."""
        uri = uri_for(path)
        with self._cond:
            before = self._seq
        version = self.versions.get(uri, 0) + 1
        self.versions[uri] = version
        if version == 1:
            doc = {"uri": uri, "languageId": language_id, "version": 1, "text": text}
            self.notify("textDocument/didOpen", {"textDocument": doc})
        else:
            self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        self.notify("textDocument/didSave", {"textDocument": {"uri": uri}, "text": text})
        return before

    def wait_diagnostics(self, path: str, after: int, wait: float) -> list[dict[str, Any]] | None:
        """Diagnostics for ``path`` published after sequence ``after``; None on timeout."""
        uri = uri_for(path)
        deadline = time.monotonic() + wait
        with self._cond:
            while True:
                seq, items = self._diags.get(uri, (0, []))
                if seq > after:
                    # servers often publish twice (syntax, then semantic): take the settled one
                    settle = min(0.15, max(deadline - time.monotonic(), 0))
                    if settle and self._cond.wait(settle):
                        continue
                    return self._diags.get(uri, (0, []))[1]
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.alive:
                    return None
                self._cond.wait(remaining)

    def close(self) -> None:
        if self.alive:
            try:
                self.request("shutdown", None, timeout=2)
                self.notify("exit", None)
            except LSPError:
                pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()  # type: ignore[union-attr]
            except OSError:
                pass


@dataclass
class Report:
    path: str
    server: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    timed_out: bool = False
    error: str = ""

    def errors(self, warnings: bool = False) -> list[Diagnostic]:
        keep = ("error", "warning") if warnings else ("error",)
        return [d for d in self.diagnostics if d.severity in keep]


class Diagnostics:
    """Language servers for one workspace, started on demand."""

    def __init__(self, executor: Any, cfg: Any):
        self.executor = executor
        self.cfg = cfg  # argus.config.LSPConfig
        self.clients: dict[str, LSPClient] = {}
        self.failed: dict[str, str] = {}

    def server_for(self, lang: str) -> list[str] | None:
        if lang in self.cfg.servers:
            argv = list(self.cfg.servers[lang])
            return argv or None
        for argv in LANGUAGES[lang]["servers"]:
            if self.executor.has_command(argv[0]):
                return list(argv)
        return None

    def client(self, lang: str) -> LSPClient | None:
        if lang in self.clients and self.clients[lang].alive:
            return self.clients[lang]
        if lang in self.failed:
            return None
        argv = self.server_for(lang)
        if argv is None:
            self.failed[lang] = "no language server installed"
            return None
        run_argv, cwd = self.executor.process_argv(argv)
        try:
            c = LSPClient(run_argv, cwd, self.executor.workdir, timeout=self.cfg.timeout)
            c.initialize()
        except (OSError, LSPError) as e:
            self.failed[lang] = f"{argv[0]}: {e}"
            return None
        self.clients[lang] = c
        return c

    def check(self, path: str, text: str) -> Report | None:
        """Diagnostics for a file that was just written; None when no server applies."""
        if not self.cfg.enabled:
            return None
        found = language_of(path)
        if found is None:
            return None
        lang, language_id = found
        c = self.client(lang)
        if c is None:
            return None
        report = Report(path, c.argv[0] if c.argv[0] != "ssh" else " ".join(c.argv[-1:]))
        try:
            before = c.update(path, text, language_id)
            items = c.wait_diagnostics(path, before, self.cfg.wait_ms / 1000)
        except LSPError as e:
            report.error = str(e)
            return report
        if items is None:
            report.timed_out = True
            return report
        for d in items:
            start = (d.get("range") or {}).get("start") or {}
            report.diagnostics.append(
                Diagnostic(
                    int(start.get("line", 0)) + 1,
                    int(start.get("character", 0)) + 1,
                    SEVERITY.get(int(d.get("severity") or 1), "error"),
                    " ".join(str(d.get("message", "")).split()),
                    str(d.get("source") or ""),
                )
            )
        return report

    def close(self) -> None:
        for c in self.clients.values():
            c.close()
        self.clients.clear()
