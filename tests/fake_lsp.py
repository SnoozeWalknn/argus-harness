#!/usr/bin/env python3
"""A tiny language server for tests: reports Python syntax errors and undefined names.

Speaks just enough LSP over stdio: initialize, didOpen / didChange / didSave (each
publishes diagnostics), shutdown and exit. It also sends the client a request
(``window/workDoneProgress/create``) to check that argus answers server requests.
Diagnostics: a SyntaxError from ``compile``, and every ``undefined_name`` in the text
as an error; ``TODO`` as a warning.
"""

from __future__ import annotations

import json
import sys


def read() -> dict | None:
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        name, _, value = line.decode().partition(":")
        if name.lower() == "content-length":
            length = int(value)
    return json.loads(sys.stdin.buffer.read(length or 0))


def send(msg: dict) -> None:
    body = json.dumps({"jsonrpc": "2.0", **msg}).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()


def diagnostics(text: str) -> list[dict]:
    out = []
    try:
        compile(text, "<file>", "exec")
    except SyntaxError as e:
        line = max((e.lineno or 1) - 1, 0)
        col = max((e.offset or 1) - 1, 0)
        out.append(_diag(line, col, 1, f"SyntaxError: {e.msg}"))
    for n, line in enumerate(text.splitlines()):
        col = line.find("undefined_name")
        if col >= 0:
            out.append(_diag(n, col, 1, '"undefined_name" is not defined'))
        col = line.find("TODO")
        if col >= 0:
            out.append(_diag(n, col, 2, "TODO left in code"))
    return out


def _diag(line: int, col: int, severity: int, message: str) -> dict:
    rng = {"start": {"line": line, "character": col}, "end": {"line": line, "character": col + 1}}
    return {"range": rng, "severity": severity, "message": message, "source": "fake-lsp"}


def main() -> None:
    docs: dict[str, str] = {}
    while True:
        msg = read()
        if msg is None:
            return
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "initialize":
            send({"id": 99, "method": "window/workDoneProgress/create", "params": {"token": "t"}})
            send({"id": msg["id"], "result": {"capabilities": {"textDocumentSync": 1}}})
        elif method in ("textDocument/didOpen", "textDocument/didChange", "textDocument/didSave"):
            doc = params["textDocument"]
            if method == "textDocument/didOpen":
                docs[doc["uri"]] = doc["text"]
            elif method == "textDocument/didChange":
                docs[doc["uri"]] = params["contentChanges"][-1]["text"]
            else:
                docs[doc["uri"]] = params.get("text", docs.get(doc["uri"], ""))
            send(
                {
                    "method": "textDocument/publishDiagnostics",
                    "params": {"uri": doc["uri"], "diagnostics": diagnostics(docs[doc["uri"]])},
                }
            )
        elif method == "shutdown":
            send({"id": msg["id"], "result": None})
        elif method == "exit":
            return
        elif "id" in msg and method:
            send({"id": msg["id"], "result": None})


if __name__ == "__main__":
    main()
