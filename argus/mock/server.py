"""A scripted stand-in for model servers and APIs.

Imitates llama-server by default: ``/v1/chat/completions`` (streaming and not),
``/tokenize``, ``/apply-template``, ``/props``, ``/health`` and ``/v1/models``.
With ``flavor=`` it imitates the discovery endpoints and chat details of
Ollama, vLLM, LM Studio, OpenRouter or a generic OpenAI-compatible server.
A step with ``replay`` sends a recorded response verbatim (fixture tests).
Responses come from a :class:`~argus.mock.script.Script`. Generation honours ``max_tokens``
and the context size (``finish_reason = "length"``), simulates KV-cache reuse
(``timings.cache_n``), stops when the client disconnects, and checks that
constrained output actually satisfies the request's schema or grammar.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
import zlib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from argus.mock.script import Script, Step, check_expect, request_protocol

TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]")
FLAVORS = ("llama_server", "ollama", "vllm", "lmstudio", "openrouter", "generic")


def tokenize(text: str) -> list[str]:
    """Deterministic stand-in tokenizer: words, punctuation and whitespace runs."""
    return TOKEN_RE.findall(text)


def token_ids(text: str) -> list[int]:
    return [zlib.crc32(t.encode()) % 150_000 for t in tokenize(text)]


QWEN_TOOLS_PREAMBLE = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
)
QWEN_TOOLS_POST = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def render_chatml(req: dict[str, Any]) -> str:
    """Approximation of Qwen's chat template (ChatML + Hermes-style tools)."""
    msgs = list(req.get("messages") or [])
    tools = req.get("tools") or []
    out = []
    system = ""
    if msgs and msgs[0].get("role") == "system":
        system = msgs.pop(0).get("content") or ""
    if tools:
        body = "\n".join(json.dumps(t, separators=(", ", ": ")) for t in tools)
        system = (system + "\n\n" if system else "") + QWEN_TOOLS_PREAMBLE + body + QWEN_TOOLS_POST
    if system:
        out.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "tool":
            out.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            )
        elif role == "assistant":
            text = content
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments", "{}")
                text += f'\n<tool_call>\n{{"name": "{fn.get("name")}", "arguments": {args}}}\n</tool_call>'
            out.append(f"<|im_start|>assistant\n{text.strip()}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    out.append("<|im_start|>assistant\n")
    return "".join(out)


class _Gen:
    """What the scripted model 'generates' for one request."""

    def __init__(self) -> None:
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []  # {"id","name","arguments": str}
        self.finish_reason = "stop"


def _repeat(text: str, n: int) -> Iterator[str]:
    pieces = tokenize(text)
    for _ in range(n):
        yield from pieces


class MockServer:
    def __init__(
        self,
        script: Script | list[Any] | None = None,
        *,
        n_ctx: int = 32768,
        host: str = "127.0.0.1",
        port: int = 0,
        model: str = "mock-qwen",
        chunk_delay: float = 0.0,
        strict: bool = True,
        flavor: str = "llama_server",
        num_ctx: int | None = None,
        verify_signatures: bool = True,
        gemini_ids: bool = False,
    ):
        if flavor not in FLAVORS:
            raise ValueError(f"unknown mock flavor {flavor!r}; choose from {', '.join(FLAVORS)}")
        self.script = script if isinstance(script, Script) else Script(script or [])
        self.flavor = flavor  # which OpenAI-compatible server to imitate
        self.num_ctx = num_ctx  # ollama: the model's num_ctx parameter, if set
        # Check opaque replay tokens the way the APIs do: Anthropic thinking signatures
        # (bound to the conversation prefix), OpenAI encrypted reasoning (bound to the
        # model), Gemini 3 thought signatures. Off when replaying real recordings.
        self.verify_signatures = verify_signatures
        self.gemini_ids = gemini_ids  # gemini: include ids in function calls (newer models do)
        self.n_ctx = n_ctx
        self.host = host
        self.model = model
        self.chunk_delay = chunk_delay
        self.strict = strict  # validate constrained output against the request
        self.requests: list[dict[str, Any]] = []  # as received (wire format)
        self.canonical_requests: list[dict[str, Any]] = []  # as OpenAI chat requests
        self.paths: list[str] = []  # request path of each generation request
        self.responses: list[dict[str, Any]] = []
        self.errors: list[str] = []  # failed expectations / invalid scripted output
        self.aborted = 0  # streams closed by the client mid-generation
        self._last_prompt: list[int] = []
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.mock = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def root(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        """Base URL for the OpenAI-compatible API of the imitated flavour."""
        return f"{self.root}/api/v1" if self.flavor == "openrouter" else f"{self.root}/v1"

    @property
    def reasoning_field(self) -> str:
        return "reasoning" if self.flavor in ("ollama", "openrouter") else "reasoning_content"

    def start(self) -> MockServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> MockServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    # -- generation ------------------------------------------------------------------------

    def prompt_tokens(self, req: dict[str, Any]) -> list[int]:
        return token_ids(render_chatml(req))

    def build(self, req: dict[str, Any], step: Step, id_prefix: str = "call_") -> _Gen:
        g = _Gen()
        proto = request_protocol(req)
        g.reasoning = list(_repeat(step.get("reasoning", ""), int(step.get("reasoning_repeat", 1))))
        calls = []
        for i, tc in enumerate(step.get("tool_calls") or []):
            args = tc.get("arguments", {})
            calls.append(
                {
                    "id": tc.get("id") or f"{id_prefix}{next(self._counter)}",
                    "name": tc["name"],
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                    "args_obj": args,
                    "index": i,
                }
            )
        if proto == "native":
            g.content = list(_repeat(step.get("content", ""), int(step.get("content_repeat", 1))))
            g.tool_calls = calls
            g.finish_reason = "tool_calls" if calls else "stop"
        else:
            text = step["raw"] if "raw" in step else self.envelope(req, step, calls)
            g.content = tokenize(text)
            if not step.get("raw"):
                self.validate_output(req, text)
        if step.get("finish_reason"):
            g.finish_reason = step["finish_reason"]
        return g

    def envelope(self, req: dict[str, Any], step: Step, calls: list[dict[str, Any]]) -> str:
        """Render a step as the JSON action envelope used by the constrained protocols."""
        spec = json.dumps(req.get("response_format") or {}) + str(req.get("grammar") or "")
        env: dict[str, Any] = {}
        if "thought" in spec:
            env["thought"] = step.get("thought", "")
        if calls:
            if len(calls) > 1:
                self.errors.append(
                    "constrained protocols take one action per turn; extra calls dropped"
                )
            c = calls[0]
            env["tool"] = c["name"]
            env["args"] = c["args_obj"] if not isinstance(c["args_obj"], str) else c["args_obj"]
            if isinstance(env["args"], str):  # malformed args under a constraint: emit raw text
                prefix = json.dumps({k: v for k, v in env.items() if k != "args"})[:-1]
                return f'{prefix}, "args": {env["args"]}}}'
        else:
            content = step.get("content", "")
            env["tool"] = "done"
            env["args"] = {"summary": content * int(step.get("content_repeat", 1))}
        return json.dumps(env, separators=(",", ":"))

    def validate_output(self, req: dict[str, Any], text: str) -> None:
        if not self.strict:
            return
        from argus.mock import constraints

        problem = constraints.check(req, text)
        if problem:
            self.errors.append(f"scripted output violates the request constraint: {problem}")

    def plan(
        self,
        req: dict[str, Any],
        step: Step,
        n_prompt: int,
        *,
        id_prefix: str = "call_",
        keep_partial_tools: bool = False,
        reasoning: bool = True,
    ) -> tuple[_Gen, list[tuple[str, Any]], int]:
        """Flatten a generation into (kind, piece) tokens, applying the output limits.

        ``keep_partial_tools`` keeps a call cut off by the limit as a partial call (as
        hosted APIs stream it) instead of surfacing it as text like llama-server;
        ``reasoning=False`` generates no reasoning (thinking switched off).
        """
        g = self.build(req, step, id_prefix)
        if not reasoning:
            g.reasoning = []
        max_tokens = req.get("max_tokens", req.get("n_predict", -1))
        if max_tokens is None or max_tokens < 0:
            max_tokens = 1 << 30
        limit = min(max_tokens, max(self.n_ctx - n_prompt, 0))
        pieces: list[tuple[str, Any]] = []
        truncated = False

        def take(kind: str, items: Any) -> bool:
            nonlocal truncated
            for it in items:
                if len(pieces) >= limit:
                    truncated = True
                    return False
                pieces.append((kind, it))
            return True

        ok = take("reasoning", g.reasoning) and take("content", g.content)
        if ok:
            for tc in g.tool_calls:
                arg_pieces = tokenize(tc["arguments"]) or [""]
                if not take("tool", [(tc, p, j == 0) for j, p in enumerate(arg_pieces)]):
                    break
        generated = len(pieces)
        if truncated and keep_partial_tools:
            g.finish_reason = "length"
            started = {id(it[0]) for kind, it in pieces if kind == "tool"}
            g.tool_calls = [tc for tc in g.tool_calls if id(tc) in started]
        elif truncated:
            g.finish_reason = "length"
            # Calls cut mid-way are not parsed; their text surfaces as content, like llama-server.
            done_calls: dict[str, str] = {}
            for kind, it in pieces:
                if kind == "tool":
                    done_calls.setdefault(it[0]["id"], "")
                    done_calls[it[0]["id"]] += it[1]
            pieces = [p for p in pieces if p[0] != "tool"]
            for cid, partial in done_calls.items():
                name = next(tc["name"] for tc in g.tool_calls if tc["id"] == cid)
                pieces.append(
                    ("content", f'<tool_call>\n{{"name": "{name}", "arguments": {partial}')
                )
            g.tool_calls = []
        return g, pieces, generated

    def record_response(
        self, g: _Gen, pieces: list[tuple[str, Any]], usage: dict[str, Any]
    ) -> dict[str, Any]:
        record = {
            "reasoning": "".join(p for k, p in pieces if k == "reasoning"),
            "content": "".join(p for k, p in pieces if k == "content"),
            "tool_calls": [
                {"name": tc["name"], "arguments": tc["arguments"]} for tc in g.tool_calls
            ],
            "finish_reason": g.finish_reason,
            "usage": usage,
        }
        with self._lock:
            self.responses.append(record)
        return record

    def cache_hit(self, prompt: list[int]) -> int:
        with self._lock:
            n = 0
            for a, b in zip(prompt, self._last_prompt, strict=False):
                if a != b:
                    break
                n += 1
            self._last_prompt = prompt
        return n


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "argus-mock/0.1"

    @property
    def mock(self) -> MockServer:
        return self.server.mock  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # silence
        pass

    # -- plumbing ----------------------------------------------------------------------------

    def _json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):  # the client gave up (timeout tests)
            with self.mock._lock:
                self.mock.aborted += 1

    def _error(
        self,
        status: int,
        message: str,
        type_: str = "server_error",
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        body = {"error": {"code": status, "message": message, "type": type_, **extra}}
        self._json(status, body, headers)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        m = self.mock
        f = m.flavor
        if path in ("/health", "/v1/health") and f in ("llama_server", "vllm"):
            self._json(200, {"status": "ok"})
        elif path in ("/props", "/v1/props") and f == "llama_server":
            self._json(
                200,
                {
                    "default_generation_settings": {"n_ctx": m.n_ctx, "params": {}},
                    "total_slots": 1,
                    "model_path": f"/models/{m.model}.gguf",
                    "chat_template": "chatml (argus mock) with tools and <think>",
                    "build_info": "argus-mock",
                },
            )
        elif path == "/api/version" and f == "ollama":
            self._json(200, {"version": "0.12.3"})
        elif path == "/api/v0/models" and f == "lmstudio":
            row = {
                "id": m.model,
                "object": "model",
                "type": "llm",
                "state": "loaded",
                "max_context_length": max(m.n_ctx, 131072),
                "loaded_context_length": m.n_ctx,
            }
            self._json(200, {"object": "list", "data": [row]})
        elif path == "/api/v1/models" and f == "openrouter":
            row = {
                "id": m.model,
                "name": m.model,
                "context_length": m.n_ctx,
                "pricing": {
                    "prompt": "0.0000002",
                    "completion": "0.0000008",
                    "input_cache_read": "0.00000005",
                },
                "supported_parameters": ["tools", "reasoning", "response_format", "temperature"],
                "top_provider": {"context_length": m.n_ctx, "max_completion_tokens": 16384},
            }
            self._json(200, {"data": [row]})
        elif path.startswith("/v1beta/models/"):
            name = path.removeprefix("/v1beta/models/")
            self._json(
                200,
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "inputTokenLimit": m.n_ctx,
                    "outputTokenLimit": 65536,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                    "thinking": True,
                },
            )
        elif path.startswith("/v1/models/"):
            # Anthropic's model object, plus Gemini's fields (its API also has a v1 path)
            name = path.removeprefix("/v1/models/")
            self._json(
                200,
                {
                    "id": name,
                    "type": "model",
                    "display_name": name,
                    "created_at": "2026-01-01T00:00:00Z",
                    "max_input_tokens": m.n_ctx,
                    "max_tokens": 64000,
                    "name": f"models/{name}",
                    "inputTokenLimit": m.n_ctx,
                    "outputTokenLimit": 65536,
                    "thinking": True,
                },
            )
        elif path in ("/v1/models", "/models") and f != "openrouter":
            row: dict[str, Any] = {"id": m.model, "object": "model", "owned_by": "argus-mock"}
            if f == "vllm":
                row["max_model_len"] = m.n_ctx
            self._json(200, {"object": "list", "data": [row]})
        else:
            self._error(404, f"no route {path}", "not_found_error")

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        f = self.mock.flavor
        try:
            body = self._body()
        except json.JSONDecodeError as e:
            self._error(400, f"invalid JSON body: {e}", "invalid_request_error")
            return
        if path in ("/tokenize", "/v1/tokenize") and f == "llama_server":
            text = body.get("content", "")
            if body.get("with_pieces"):
                self._json(
                    200,
                    {
                        "tokens": [
                            {"id": i, "piece": p}
                            for i, p in zip(token_ids(text), tokenize(text), strict=True)
                        ]
                    },
                )
            else:
                self._json(200, {"tokens": token_ids(text)})
        elif path in ("/apply-template", "/v1/apply-template") and f == "llama_server":
            self._json(200, {"prompt": render_chatml(body)})
        elif path == "/api/show" and f == "ollama":
            m = self.mock
            params = f"num_ctx                        {m.num_ctx}" if m.num_ctx else ""
            self._json(
                200,
                {
                    "parameters": params,
                    "model_info": {
                        "general.architecture": "qwen3",
                        "qwen3.context_length": m.n_ctx,
                    },
                    "capabilities": ["completion", "tools", "thinking"],
                },
            )
        elif path in ("/v1/chat/completions", "/chat/completions", "/api/v1/chat/completions"):
            self._chat(body)
        elif path == "/v1/messages":
            self._anthropic(body)
        elif path == "/v1/responses":
            self._responses(body)
        elif path.startswith(("/v1beta/models/", "/v1/models/")) and ":" in path:
            model, _, method = path.split("/models/", 1)[1].partition(":")
            self._gemini(model, method, body)
        elif path == "/v1/messages/count_tokens":
            from argus.mock import anthropic as fmt

            n = len(self.mock.prompt_tokens(fmt.to_canonical(body)))
            self._json(200, {"input_tokens": n})
        else:
            self._error(404, f"no route {path}", "not_found_error")

    # -- generation, shared by every wire format ------------------------------------------------

    def _take_step(self, req: dict[str, Any], canonical: dict[str, Any]) -> Step | None:
        """Record the request, then take the next step unless the prompt overflows.

        Returns None when a response (an error) has already been sent.
        """
        m = self.mock
        self._log_request(req, canonical)
        step = m.script.next(canonical, m)
        if step.get("replay"):
            return step
        if step.get("expect"):
            problems = check_expect(step["expect"], canonical)
            if problems:
                msg = "mock expectation failed: " + "; ".join(problems)
                m.errors.append(msg)
                self._error(500, msg, "mock_error")
                return None
        if step.get("delay"):
            time.sleep(float(step["delay"]))
        return step

    def _log_request(self, req: dict[str, Any], canonical: dict[str, Any]) -> None:
        m = self.mock
        with m._lock:
            m.requests.append(req)
            m.canonical_requests.append(canonical)
            m.paths.append(self.path.split("?")[0])

    def _replay(self, rec: dict[str, Any]) -> None:
        """Send a recorded response verbatim: a JSON body or a list of SSE events."""
        status = int(rec.get("status", 200))
        if "sse" not in rec:
            body = rec.get("json")
            data = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for k, v in (rec.get("headers") or {}).items():
                if k.lower() not in ("content-type", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in rec["sse"]:
                if "comment" in ev:
                    self.wfile.write(f": {ev['comment']}\n\n".encode())
                    continue
                data = ev.get("data")
                text = data if isinstance(data, str) else json.dumps(data)
                head = f"event: {ev['event']}\n" if ev.get("event") else ""
                self.wfile.write(f"{head}data: {text}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with self.mock._lock:
                self.mock.aborted += 1

    def _overflow(self, n_prompt: int) -> None:
        m = self.mock
        if m.flavor in ("vllm", "openrouter", "generic", "lmstudio"):
            self._error(
                400,
                f"This model's maximum context length is {m.n_ctx} tokens. However, your "
                f"messages resulted in {n_prompt} tokens. Please reduce the length of the messages.",
                "invalid_request_error",
                param="messages",
            )
            return
        self._error(
            400,
            "the request exceeds the available context size, try increasing it",
            "exceed_context_size_error",
            n_prompt_tokens=n_prompt,
            n_ctx=m.n_ctx,
        )

    # -- chat completions ----------------------------------------------------------------------

    def _chat(self, req: dict[str, Any]) -> None:
        m = self.mock
        # An oversized prompt is rejected before the model "generates" (no step consumed).
        prompt = m.prompt_tokens(req)
        if len(prompt) > m.n_ctx:
            self._log_request(req, req)
            self._overflow(len(prompt))
            return
        step = self._take_step(req, req)
        if step is None:
            return
        if step.get("replay"):
            self._replay(step["replay"])
            return
        if step.get("error"):
            e = step["error"]
            self._error(
                int(e.get("status", 500)),
                e.get("message", "mock error"),
                e.get("type", "server_error"),
                headers=e.get("headers"),
            )
            return
        cache_n = m.cache_hit(prompt)
        g, pieces, n_gen = m.plan(req, step, len(prompt))
        timings = {
            "cache_n": cache_n,
            "prompt_n": len(prompt) - cache_n,
            "prompt_ms": (len(prompt) - cache_n) * 0.05,
            "prompt_per_second": 20000.0,
            "predicted_n": n_gen,
            "predicted_ms": n_gen * 1.0,
            "predicted_per_second": 1000.0,
        }
        usage: dict[str, Any] = {
            "prompt_tokens": len(prompt),
            "completion_tokens": n_gen,
            "total_tokens": len(prompt) + n_gen,
            "prompt_tokens_details": {"cached_tokens": cache_n},
        }
        if m.flavor == "openrouter":
            usage["cost"] = round(len(prompt) * 2e-7 + n_gen * 8e-7, 8)
        if m.flavor != "llama_server":
            timings = {}
        m.record_response(g, pieces, usage)
        if req.get("stream"):
            self._stream(req, step, g, pieces, usage, timings)
            return
        record = m.responses[-1]
        message: dict[str, Any] = {"role": "assistant", "content": record["content"] or None}
        if record["reasoning"]:
            message[m.reasoning_field] = record["reasoning"]
        if g.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in g.tool_calls
            ]
        out: dict[str, Any] = {
            "id": f"chatcmpl-{next(m._counter)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", m.model),
            "choices": [{"index": 0, "message": message, "finish_reason": g.finish_reason}],
            "usage": usage,
        }
        if timings:
            out["timings"] = timings
        self._json(200, out)

    def _stream(
        self,
        req: dict[str, Any],
        step: Step,
        g: _Gen,
        pieces: list[tuple[str, Any]],
        usage: dict[str, Any],
        timings: dict[str, Any],
    ) -> None:
        m = self.mock
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        cid = f"chatcmpl-{next(m._counter)}"
        delay = float(step.get("chunk_delay", m.chunk_delay))

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.get("model", m.model),
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }

        def send(obj: Any) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        try:
            if m.flavor == "openrouter":
                self.wfile.write(b": OPENROUTER PROCESSING\n\n")
            send(chunk({"role": "assistant", "content": None}))
            for kind, piece in pieces:
                if delay:
                    time.sleep(delay)
                if kind == "reasoning":
                    send(chunk({m.reasoning_field: piece}))
                elif kind == "content":
                    send(chunk({"content": piece}))
                else:
                    tc, text, first = piece
                    fn: dict[str, Any] = {"arguments": text}
                    entry: dict[str, Any] = {"index": tc["index"], "function": fn}
                    if first:
                        entry.update(id=tc["id"], type="function")
                        fn["name"] = tc["name"]
                    send(chunk({"tool_calls": [entry]}))
            extra = {"timings": timings} if timings else {}
            send(chunk({}, g.finish_reason, **extra))
            if (req.get("stream_options") or {}).get("include_usage"):
                send(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": usage,
                        **extra,
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with m._lock:
                m.aborted += 1

    # -- Anthropic Messages API ----------------------------------------------------------------

    def _anthropic_error(self, status: int, kind: str, message: str) -> None:
        from argus.mock import anthropic as fmt

        self._json(status, fmt.error_body(kind, message))

    def _anthropic(self, req: dict[str, Any]) -> None:
        from argus.mock import anthropic as fmt

        m = self.mock
        canon = fmt.to_canonical(req)
        if not self.headers.get("x-api-key"):
            self._log_request(req, canon)
            self._anthropic_error(401, "authentication_error", "x-api-key header is required")
            return
        if not self.headers.get("anthropic-version"):
            self._log_request(req, canon)
            self._anthropic_error(400, "invalid_request_error", "anthropic-version header required")
            return
        prompt = m.prompt_tokens(canon)
        problems = fmt.validate(req) if m.strict else []
        bad_sig = fmt.check_signatures(req) if m.verify_signatures else None
        if problems or bad_sig or len(prompt) > m.n_ctx:
            self._log_request(req, canon)
            if problems:
                m.errors.append("invalid anthropic request: " + "; ".join(problems))
                self._anthropic_error(400, "invalid_request_error", "; ".join(problems))
            elif bad_sig:
                self._anthropic_error(400, "invalid_request_error", bad_sig)
            else:
                self._anthropic_error(
                    400,
                    "invalid_request_error",
                    f"prompt is too long: {len(prompt)} tokens > {m.n_ctx} maximum",
                )
            return
        step = self._take_step(req, canon)
        if step is None:
            return
        if step.get("replay"):
            self._replay(step["replay"])
            return
        if step.get("error"):
            e = step["error"]
            status = int(e.get("status", 500))
            kind = e.get("type") or {429: "rate_limit_error", 529: "overloaded_error"}.get(
                status, "api_error"
            )
            body = fmt.error_body(kind, e.get("message", "mock error"))
            self._json(status, body, e.get("headers"))
            return
        thinking = req.get("thinking") or {}
        think_on = thinking.get("type") in ("adaptive", "enabled")
        show = thinking.get("type") == "enabled" or thinking.get("display") == "summarized"
        cache_n = m.cache_hit(prompt)
        g, pieces, n_gen = m.plan(
            canon,
            step,
            len(prompt),
            id_prefix="toolu_mock_",
            keep_partial_tools=True,
            reasoning=think_on,
        )
        blocks = fmt.group_blocks(pieces, show)
        shown = "".join("".join(b["pieces"]) for b in blocks if b["type"] == "thinking")
        signature = fmt.sign(req, shown)
        usage = fmt.usage(req, len(prompt), cache_n, n_gen)
        stop = fmt.stop_reason(step, g.finish_reason)
        details = fmt.stop_details(step)
        m.record_response(g, pieces, usage)
        message = {
            "id": f"msg_mock_{next(m._counter)}",
            "type": "message",
            "role": "assistant",
            "model": req.get("model", m.model),
            "content": [fmt.final_block(b, signature) for b in blocks],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": usage,
        }
        if details:
            message["stop_details"] = details
        if not req.get("stream"):
            self._json(200, message)
            return
        self._anthropic_stream(step, message, blocks, signature, details)

    def _anthropic_stream(
        self,
        step: Step,
        message: dict[str, Any],
        blocks: list[dict[str, Any]],
        signature: str,
        details: dict[str, Any] | None,
    ) -> None:
        m = self.mock
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        delay = float(step.get("chunk_delay", m.chunk_delay))

        def send(event: str, data: dict[str, Any]) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        usage = message["usage"]
        start = {**message, "content": [], "stop_reason": None}
        start["usage"] = {**usage, "output_tokens": 1}
        start.pop("stop_details", None)
        try:
            send("message_start", {"type": "message_start", "message": start})
            send("ping", {"type": "ping"})
            for i, b in enumerate(blocks):
                if b["type"] == "thinking":
                    head: dict[str, Any] = {"type": "thinking", "thinking": "", "signature": ""}
                elif b["type"] == "text":
                    head = {"type": "text", "text": ""}
                else:
                    head = {"type": "tool_use", "id": b["tc"]["id"], "name": b["tc"]["name"]}
                    head["input"] = {}
                send(
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": head},
                )
                for piece in b["pieces"]:
                    if delay:
                        time.sleep(delay)
                    if b["type"] == "thinking":
                        if not piece:
                            continue
                        delta = {"type": "thinking_delta", "thinking": piece}
                    elif b["type"] == "text":
                        delta = {"type": "text_delta", "text": piece}
                    else:
                        delta = {"type": "input_json_delta", "partial_json": piece}
                    send(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": i, "delta": delta},
                    )
                if b["type"] == "thinking":
                    sig = {"type": "signature_delta", "signature": signature}
                    send(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": i, "delta": sig},
                    )
                send("content_block_stop", {"type": "content_block_stop", "index": i})
            if step.get("stream_error"):
                e = step["stream_error"]
                err = {
                    "type": e.get("type", "overloaded_error"),
                    "message": e.get("message", "Overloaded"),
                }
                send("error", {"type": "error", "error": err})
                return
            delta: dict[str, Any] = {"stop_reason": message["stop_reason"], "stop_sequence": None}
            if details:
                delta["stop_details"] = details
            send(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": delta,
                    "usage": {"output_tokens": usage["output_tokens"]},
                },
            )
            send("message_stop", {"type": "message_stop"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with m._lock:
                m.aborted += 1

    # -- OpenAI Responses API ------------------------------------------------------------------

    def _responses(self, req: dict[str, Any]) -> None:
        from argus.mock import openai_responses as fmt

        m = self.mock
        canon = fmt.to_canonical(req)
        auth = self.headers.get("authorization") or ""
        if not auth.startswith("Bearer ") or len(auth) <= 7:
            self._log_request(req, canon)
            body = fmt.error_body(None, "You didn't provide an API key.")
            self._json(401, body)
            return
        prompt = m.prompt_tokens(canon)
        problem = fmt.validate(req, m.verify_signatures) if m.strict else None
        if problem or len(prompt) > m.n_ctx:
            self._log_request(req, canon)
            if problem:
                status, code, message = problem
                if code != "invalid_encrypted_content":
                    m.errors.append(f"invalid responses request: {message}")
                self._json(status, fmt.error_body(code, message))
            else:
                message = (
                    "Your input exceeds the context window of this model. "
                    "Please adjust your input and try again."
                )
                self._json(400, fmt.error_body("context_length_exceeded", message))
            return
        step = self._take_step(req, canon)
        if step is None:
            return
        if step.get("replay"):
            self._replay(step["replay"])
            return
        if step.get("error"):
            e = step["error"]
            status = int(e.get("status", 500))
            kind = e.get("type") or ("rate_limit_exceeded" if status == 429 else "server_error")
            self._json(
                status, fmt.error_body(kind, e.get("message", "mock error"), kind), e.get("headers")
            )
            return
        reasoning_cfg = req.get("reasoning")
        think_on = reasoning_cfg is not None and reasoning_cfg.get("effort") != "minimal"
        cache_n = m.cache_hit(prompt)
        g, pieces, n_gen = m.plan(
            canon,
            step,
            len(prompt),
            id_prefix="call_mock_",
            keep_partial_tools=True,
            reasoning=think_on,
        )
        output = fmt.build_output(req, step, g, pieces, m._counter)
        n_reasoning = sum(1 for k, _ in pieces if k == "reasoning")
        usage = fmt.usage(len(prompt), cache_n, n_gen, n_reasoning)
        base = fmt.response_object(
            f"resp_mock_{next(m._counter)}", req, output, usage, g.finish_reason == "length"
        )
        m.record_response(g, pieces, usage)
        if not req.get("stream"):
            self._json(200, base)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        delay = float(step.get("chunk_delay", m.chunk_delay))
        try:
            for name, data in fmt.stream_events(base, output, pieces):
                if step.get("stream_error") and name in (
                    "response.completed",
                    "response.incomplete",
                ):
                    err = {
                        "type": "error",
                        "code": "server_error",
                        "message": "The server had an error",
                    }
                    self.wfile.write(f"event: error\ndata: {json.dumps(err)}\n\n".encode())
                    return
                if delay and name.endswith(".delta"):
                    time.sleep(delay)
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with m._lock:
                m.aborted += 1

    # -- Gemini API ------------------------------------------------------------------------------

    def _gemini(self, model: str, method: str, req: dict[str, Any]) -> None:
        from argus.mock import gemini as fmt

        m = self.mock
        if method == "countTokens":
            inner = req.get("generateContentRequest") or req
            n = len(m.prompt_tokens(fmt.to_canonical(model, inner)))
            self._json(200, {"totalTokens": n})
            return
        canon = fmt.to_canonical(model, req)
        if not self.headers.get("x-goog-api-key"):
            self._log_request(req, canon)
            self._json(403, fmt.error_body(403, "PERMISSION_DENIED", "API key required"))
            return
        prompt = m.prompt_tokens(canon)
        problem = fmt.validate(model, req, m.verify_signatures) if m.strict else None
        if problem or len(prompt) > m.n_ctx:
            self._log_request(req, canon)
            if problem:
                if "thought_signature" not in problem:
                    m.errors.append(f"invalid gemini request: {problem}")
                self._json(400, fmt.error_body(400, "INVALID_ARGUMENT", problem))
            else:
                message = (
                    f"The input token count ({len(prompt)}) exceeds the maximum number of "
                    f"tokens allowed ({m.n_ctx})."
                )
                self._json(400, fmt.error_body(400, "INVALID_ARGUMENT", message))
            return
        step = self._take_step(req, canon)
        if step is None:
            return
        if step.get("replay"):
            self._replay(step["replay"])
            return
        if step.get("error"):
            e = step["error"]
            status = int(e.get("status", 500))
            kind = {429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE"}.get(status, "INTERNAL")
            body = fmt.error_body(status, e.get("type") or kind, e.get("message", "mock error"))
            self._json(status, body, e.get("headers"))
            return
        tc = (req.get("generationConfig") or {}).get("thinkingConfig") or {}
        think_on = tc.get("thinkingBudget") != 0
        show = bool(tc.get("includeThoughts"))
        cache_n = m.cache_hit(prompt)
        g, pieces, n_gen = m.plan(
            canon,
            step,
            len(prompt),
            id_prefix="gcall_",
            keep_partial_tools=True,
            reasoning=think_on,
        )
        finish = fmt.finish_reason(step, g.finish_reason)
        if fmt.harm_blocked(step, req):
            finish = "SAFETY"
        if finish == "SAFETY":
            pieces, g.tool_calls = [], []
        n_thoughts = sum(1 for k, _ in pieces if k == "reasoning")
        usage = fmt.usage(len(prompt), cache_n, len(pieces) - n_thoughts, n_thoughts)
        chunks = fmt.chunks(model, step, pieces, g.tool_calls, finish, usage, show, m.gemini_ids)
        m.record_response(g, pieces, usage)
        if method != "streamGenerateContent":
            self._json(200, fmt.merged(chunks))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        delay = float(step.get("chunk_delay", m.chunk_delay))
        try:
            for ch in chunks:
                if delay:
                    time.sleep(delay)
                self.wfile.write(f"data: {json.dumps(ch)}\r\n\r\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with m._lock:
                m.aborted += 1
