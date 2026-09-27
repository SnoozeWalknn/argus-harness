"""OpenAI-compatible chat completions: llama-server, Ollama, vLLM, LM Studio, OpenRouter.

All of them take ``POST /v1/chat/completions`` and stream ``chat.completion.chunk``
events; they differ in the details, which this adapter handles per *flavour*:

============  ==================================================================
llama_server  ``/props`` (n_ctx), ``/tokenize``, ``/apply-template``, GBNF ``grammar``,
              ``timings``, thinking via ``chat_template_kwargs.enable_thinking``
ollama        ``/api/show`` (context length, capabilities), ``reasoning`` field,
              thinking via ``think``
vllm          ``max_model_len`` in ``/v1/models``, GBNF via ``guided_grammar``
lmstudio      ``/api/v0/models`` (max and loaded context length)
openrouter    ``/api/v1/models`` (context length, pricing), ``reasoning`` field and
              request parameter, SSE comment keep-alives
generic       anything else that speaks the protocol
============  ==================================================================

The flavour is detected by probing those endpoints unless it is configured.
Reasoning is read from ``reasoning_content`` or ``reasoning``, whichever the
server uses.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from argus.providers.base import (
    Completion,
    Detected,
    LLMError,
    Monitor,
    NotSupported,
    Provider,
    RawToolCall,
    StopGeneration,
    mark_ttft,
    root_url,
    run_monitors,
    stream_guard,
    text_of,
)

FLAVORS = ("llama_server", "ollama", "vllm", "lmstudio", "openrouter", "generic")
LOCAL_PORTS = {"llama_server": 8080, "ollama": 11434, "lmstudio": 1234, "vllm": 8000}
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty")
FINISH = {"tool_calls": "tool_calls", "function_call": "tool_calls", "content_filter": "refusal"}


def default_base_url(flavor: str) -> str:
    if flavor == "openrouter":
        return "https://openrouter.ai/api/v1"
    return f"http://127.0.0.1:{LOCAL_PORTS.get(flavor, 8080)}/v1"


def normalise_finish(reason: str | None) -> str | None:
    if reason is None:
        return None
    return FINISH.get(reason, reason)


def merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive user (or system) messages; some templates reject them."""
    out: list[dict[str, Any]] = []
    for m in messages:
        prev = out[-1] if out else None
        if (
            prev is not None
            and m.get("role") == prev.get("role")
            and m.get("role") in ("user", "system")
            and isinstance(prev.get("content"), str)
            and isinstance(m.get("content"), str)
        ):
            out[-1] = {**prev, "content": f"{prev['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


def fold_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For templates without a system role: prepend system text to the first user turn."""
    system = [text_of(m.get("content")) for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not system:
        return rest
    prefix = "\n\n".join(system)
    for i, m in enumerate(rest):
        if m.get("role") == "user":
            rest[i] = {**m, "content": f"{prefix}\n\n{text_of(m.get('content'))}"}
            return rest
    return [{"role": "user", "content": prefix}, *rest]


class OpenAICompatProvider(Provider):
    kind = "openai_compat"

    def __init__(self, base_url: str = "", *args: Any, **kwargs: Any):
        options = kwargs.get("options")
        flavor = getattr(options, "flavor", "auto") if options else "auto"
        self.default_base_url = default_base_url(flavor if flavor in FLAVORS else "llama_server")
        super().__init__(base_url, *args, **kwargs)
        self.root = root_url(self.base_url)

    @property
    def token_chunks(self) -> bool:  # type: ignore[override]
        # Local servers stream one token per chunk; hosted routers batch.
        return self.flavor_name() != "openrouter"

    def describe(self) -> str:
        return f"{self.flavor_name() or self.kind} {self.base_url}"

    # -- flavour -----------------------------------------------------------------------------

    def flavor_name(self) -> str:
        f = self.opts.flavor
        if f in FLAVORS:
            return f
        if "openrouter.ai" in (urlsplit(self.base_url).hostname or ""):
            return "openrouter"
        return self.detect().flavor or "generic"

    def _detect(self) -> Detected:
        forced = self.opts.flavor if self.opts.flavor in FLAVORS else None
        if forced is None and "openrouter.ai" in (urlsplit(self.base_url).hostname or ""):
            forced = "openrouter"
        probes = {
            "llama_server": self._probe_llama,
            "ollama": self._probe_ollama,
            "lmstudio": self._probe_lmstudio,
            "vllm": self._probe_models,
            "openrouter": self._probe_models,
            "generic": self._probe_models,
        }
        order = [forced] if forced else ["llama_server", "ollama", "lmstudio", "generic"]
        for name in order:
            try:
                d = probes[name]()
            except httpx.ConnectError:
                raise
            except (httpx.HTTPError, LLMError, ValueError, KeyError, TypeError, AttributeError):
                d = None
            if d is not None:
                if forced and not d.flavor:
                    d.flavor = forced
                return d
        return Detected(flavor=forced or "generic")

    def detect(self, refresh: bool = False) -> Detected:
        if self._detected is None or refresh:
            try:
                self._detected = self._detect()
            except httpx.HTTPError as e:  # unreachable: say so, but try again next time
                d = Detected(notes=[f"detection failed: {type(e).__name__}: {e}"])
                if self.opts.flavor in FLAVORS:
                    d.flavor = self.opts.flavor
                return d
        return self._detected

    def _probe_llama(self) -> Detected | None:
        p = self.get_json(f"{self.root}/props")
        if not isinstance(p, dict) or not ("default_generation_settings" in p or "n_ctx" in p):
            return None
        dgs = p.get("default_generation_settings") or {}
        template = str(p.get("chat_template") or "")
        caps = {"grammar", "json_schema"}
        if "tool" in template:
            caps.add("tools")
        if "think" in template or "reasoning" in template:
            caps.add("thinking")
        name = p.get("model_alias") or str(p.get("model_path") or "").rsplit("/", 1)[-1]
        return Detected(
            flavor="llama_server",
            context_window=int(dgs.get("n_ctx") or p.get("n_ctx") or 0),
            models=[name] if name else [],
            capabilities=caps,
            raw={"props": {k: p.get(k) for k in ("model_path", "model_alias", "build_info")}},
        )

    def _probe_ollama(self) -> Detected | None:
        v = self.get_json(f"{self.root}/api/version")
        if not isinstance(v, dict) or "version" not in v:
            return None
        d = Detected(flavor="ollama", raw={"version": v["version"]})
        if not self.model:
            return d
        show = self.post_json(f"{self.root}/api/show", {"model": self.model})
        info = show.get("model_info") or {}
        trained = next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)
        m = re.search(r"^num_ctx\s+(\d+)", str(show.get("parameters") or ""), re.M)
        d.models = [self.model]
        d.capabilities = set(show.get("capabilities") or []) | {"json_schema"}
        if m:
            d.context_window = int(m.group(1))
        else:
            # Ollama serves num_ctx tokens and silently drops the rest of a longer prompt.
            d.context_window = min(trained or 4096, 4096)
            d.notes.append(
                f"no num_ctx set: Ollama serves 4096 tokens by default (model supports "
                f"{trained or 'unknown'}); set num_ctx or OLLAMA_CONTEXT_LENGTH"
            )
        d.raw["trained_context"] = trained
        return d

    def _probe_lmstudio(self) -> Detected | None:
        data = self.get_json(f"{self.root}/api/v0/models")
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows or "max_context_length" not in rows[0]:
            return None
        row = next((r for r in rows if r.get("id") == self.model), rows[0])
        return Detected(
            flavor="lmstudio",
            context_window=int(row.get("loaded_context_length") or row.get("max_context_length")),
            models=[r.get("id", "") for r in rows],
            capabilities={"tools", "json_schema"},
            raw={"model": row},
        )

    def _probe_models(self) -> Detected | None:
        data = self.get_json(f"{self.base_url}/models")
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return None
        ids = [str(r.get("id", "")) for r in rows]
        row = next((r for r in rows if r.get("id") == self.model), rows[0] if rows else {})
        if "max_model_len" in row:
            return Detected(
                flavor="vllm",
                context_window=int(row["max_model_len"]),
                models=ids,
                capabilities={"tools", "json_schema", "grammar"},
            )
        if "context_length" in row and "pricing" in row:
            price = row.get("pricing") or {}

            def per_million(key: str) -> float:
                try:
                    return round(float(price.get(key) or 0) * 1_000_000, 6)
                except (TypeError, ValueError):
                    return 0.0

            params = set(row.get("supported_parameters") or [])
            caps = {"json_schema"} if params & {"structured_outputs", "response_format"} else set()
            caps |= {"tools"} if "tools" in params else set()
            caps |= {"thinking"} if params & {"reasoning", "include_reasoning"} else set()
            top = row.get("top_provider") or {}
            return Detected(
                flavor="openrouter",
                context_window=int(row.get("context_length") or 0),
                max_output=int(top.get("max_completion_tokens") or 0),
                models=ids,
                capabilities=caps,
                pricing={
                    "input": per_million("prompt"),
                    "output": per_million("completion"),
                    "cache_read": per_million("input_cache_read"),
                    "cache_write": per_million("input_cache_write"),
                },
            )
        return Detected(flavor="generic", models=ids, capabilities={"tools"})

    # -- requests --------------------------------------------------------------------------

    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def wire_body(self, body: dict[str, Any], stream: bool) -> dict[str, Any]:
        flavor = self.flavor_name()
        q = self.opts.quirks
        out = {k: v for k, v in body.items() if k not in ("thinking", "cache")}
        msgs = [{k: v for k, v in m.items() if k != "replay"} for m in body.get("messages") or []]
        if "no_system_role" in q:
            msgs = fold_system(msgs)
        if "merge_same_role" in q:
            msgs = merge_same_role(msgs)
        if "strip_reasoning_history" in q:
            msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]
        out["messages"] = msgs

        thinking = body.get("thinking") or {}
        enabled = thinking.get("enabled")
        effort = thinking.get("effort") or self.opts.effort
        budget = thinking.get("budget") or self.opts.thinking_budget
        if flavor == "openrouter":
            reasoning: dict[str, Any] = {}
            if enabled is not None:
                reasoning["enabled"] = bool(enabled)
            if effort and enabled is not False:
                reasoning["effort"] = effort
            elif budget and enabled is not False:
                reasoning["max_tokens"] = budget
            if reasoning:
                out["reasoning"] = reasoning
        elif flavor == "ollama":
            if enabled is not None:
                out["think"] = bool(enabled)
            if effort and enabled is not False:
                out["reasoning_effort"] = effort
        else:
            if enabled is not None:
                out["chat_template_kwargs"] = {
                    **(out.get("chat_template_kwargs") or {}),
                    "enable_thinking": bool(enabled),
                }
            if effort and enabled is not False:
                if flavor == "llama_server":
                    out["chat_template_kwargs"] = {
                        **(out.get("chat_template_kwargs") or {}),
                        "reasoning_effort": effort,
                    }
                else:
                    out["reasoning_effort"] = effort

        if "grammar" in out and flavor == "vllm":
            out["guided_grammar"] = out.pop("grammar")
        if "no_sampling" in q:
            for key in SAMPLING_KEYS:
                out.pop(key, None)
        if "max_completion_tokens" in q and "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        if "no_parallel_tools" in q:
            out.pop("parallel_tool_calls", None)
        if stream:
            out["stream"] = True
            out["stream_options"] = {"include_usage": True}
        else:
            out["stream"] = False
            out.pop("stream_options", None)
        return out

    def _chat(self, body: dict[str, Any], stream: bool, monitors: list[Monitor]) -> Completion:
        wire = self.wire_body(body, stream)
        with self.exchange("/chat/completions", body, wire, stream) as ex:
            c = self._stream(wire, monitors, ex) if stream else self._once(wire, ex)
            ex.completion = c
            return c

    def _once(self, wire: dict[str, Any], ex: Any) -> Completion:
        t0 = time.perf_counter()
        data = self.post_json(self.chat_url(), wire, ex)
        c = Completion(raw=data)
        c.total_ms = (time.perf_counter() - t0) * 1000
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        c.content = msg.get("content") or ""
        c.reasoning = reasoning_of(msg)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            c.tool_calls.append(RawToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        c.stop_raw = choice.get("finish_reason")
        c.finish_reason = normalise_finish(c.stop_raw)
        if msg.get("refusal"):
            c.refusal = msg["refusal"]
            c.finish_reason = "refusal"
        c.usage = data.get("usage") or {}
        c.timings = data.get("timings") or {}
        return c

    def _stream(self, wire: dict[str, Any], monitors: list[Monitor], ex: Any) -> Completion:
        c = Completion(token_chunks=self.token_chunks)
        calls: dict[int, dict[str, str]] = {}
        t0 = time.perf_counter()
        got = [False]
        with (
            stream_guard(lambda: got[0], self.http.timeout.read),
            self.stream_events(self.chat_url(), wire, ex) as events,
        ):
            try:
                for ev in events:
                    if ev.event == "error":
                        raise LLMError(f"stream error: {ev.data.strip()}")
                    if ev.data.strip() == "[DONE]":
                        break
                    chunk = ev.json()
                    got[0] = True
                    if "error" in chunk:
                        raise LLMError(f"stream error: {chunk['error']}", body=chunk)
                    self._apply_chunk(c, calls, chunk, t0, monitors)
            except StopGeneration as stop:
                c.aborted = stop.reason
                c.abort_detail = stop.detail
        c.total_ms = (time.perf_counter() - t0) * 1000
        for idx in sorted(calls):
            tc = calls[idx]
            c.tool_calls.append(RawToolCall(tc["id"] or f"call_{idx}", tc["name"], tc["arguments"]))
        c.finish_reason = normalise_finish(c.stop_raw)
        if c.refusal:
            c.finish_reason = "refusal"
        c.raw = {
            "stream": True,
            "message": {
                "content": c.content,
                "reasoning_content": c.reasoning,
                "tool_calls": [tc.__dict__ for tc in c.tool_calls],
            },
            "finish_reason": c.stop_raw,
            "usage": c.usage,
            "timings": c.timings,
            "aborted": c.aborted,
        }
        return c

    @staticmethod
    def _apply_chunk(
        c: Completion,
        calls: dict[int, dict[str, str]],
        chunk: dict[str, Any],
        t0: float,
        monitors: list[Monitor],
    ) -> None:
        if chunk.get("usage"):
            c.usage = chunk["usage"]
        if chunk.get("timings"):
            c.timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                c.stop_raw = choice["finish_reason"]
            delta = choice.get("delta") or {}
            events: list[tuple[str, str]] = []
            r = reasoning_of(delta)
            if r:
                c.reasoning += r
                c.reasoning_chunks += 1
                events.append(("reasoning", r))
            if delta.get("content"):
                c.content += delta["content"]
                c.content_chunks += 1
                events.append(("content", delta["content"]))
            if delta.get("refusal"):
                c.refusal = (c.refusal or "") + delta["refusal"]
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(calls))
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                args = fn.get("arguments")
                if args:
                    if not isinstance(args, str):  # some servers send the object whole
                        args = json.dumps(args)
                    slot["arguments"] += args
                c.content_chunks += 1
                events.append(("tool", args or fn.get("name") or ""))
            if events:
                mark_ttft(c, t0)
            run_monitors(c, events, monitors)

    # -- llama-server native endpoints ----------------------------------------------------------

    def _native(self) -> None:
        if self.flavor_name() not in ("llama_server", "generic"):
            raise NotSupported(f"{self.flavor_name()} has no /tokenize or /apply-template")

    def tokenize(self, text: str) -> list[int]:
        self._native()
        resp = self.http.post(f"{self.root}/tokenize", json={"content": text, "add_special": False})
        self.raise_for(resp)
        return list(resp.json().get("tokens") or [])

    def apply_template(self, body: dict[str, Any]) -> str:
        self._native()
        wire = self.wire_body(body, False)
        wire.pop("stream", None)
        resp = self.http.post(f"{self.root}/apply-template", json=wire)
        self.raise_for(resp)
        return resp.json()["prompt"]

    def props(self) -> dict[str, Any]:
        return self.get_json(f"{self.root}/props")

    def health(self) -> bool:
        try:
            if self.http.get(f"{self.root}/health").status_code == 200:
                return True
            return self.http.get(f"{self.base_url}/models").status_code == 200
        except httpx.HTTPError:
            return False


def reasoning_of(obj: dict[str, Any]) -> str:
    value = obj.get("reasoning_content")
    if not value:
        value = obj.get("reasoning")
    return value if isinstance(value, str) else ""
