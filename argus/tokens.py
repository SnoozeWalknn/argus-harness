"""Token counting and prompt-overhead measurement.

Exact numbers come from llama-server itself: ``/apply-template`` renders a
request with the model's chat template (tools included) and ``/tokenize``
counts it. Differencing renders with and without each part attributes the
overhead to the system prompt, each tool schema and the template framing.
Without a server, counts fall back to a character-based estimate and say so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from argus.providers.base import NotSupported, Provider

if TYPE_CHECKING:
    from argus.agent import Agent

CHARS_PER_TOKEN = 3.7  # rough average for code and English with Qwen's tokenizer


def estimate(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


class TokenCounter:
    """Counts tokens as exactly as the provider allows.

    ``server``    llama-server's ``/tokenize`` and ``/apply-template``
    ``api``       the provider counts whole requests (Anthropic ``count_tokens``,
                  Gemini ``countTokens``); single texts are estimated unless ``exact``
    ``estimate``  characters / 3.7 on an approximate chat template
    """

    def __init__(self, llm: Provider | None):
        self.llm = llm
        self.method = "server" if llm is not None else "estimate"
        self.error: str | None = None
        self._cache: dict[tuple[str, bool], int] = {}
        self._api_base: int | None = None

    def _degrade(self, e: Exception) -> None:
        if isinstance(e, NotSupported) and self.method == "server":
            self.method = "api"  # no tokenizer; the provider may still count requests
            return
        self.method = "estimate"
        self.error = f"{type(e).__name__}: {e}"

    def count(self, text: str, exact: bool = False) -> int:
        if not text:
            return 0
        key = (text, exact and self.method == "api")
        if key in self._cache:
            return self._cache[key]
        n = None
        if self.method == "server":
            try:
                n = len(self.llm.tokenize(text))  # type: ignore[union-attr]
            except Exception as e:
                self._degrade(e)
        if n is None and self.method == "api" and exact:
            n = self._api_count(text)
        if n is None:
            n = estimate(text)
        self._cache[(text, exact and self.method == "api")] = n
        return n

    def _api_count(self, text: str) -> int | None:
        def req(t: str) -> int | None:
            return self._api_prompt({"messages": [{"role": "user", "content": t}]})

        if self._api_base is None:
            self._api_base = req("x")
        n = req(text)
        if n is None or self._api_base is None:
            return None
        return max(n - self._api_base + 1, 0)

    def _api_prompt(self, body: dict[str, Any]) -> int | None:
        try:
            n = self.llm.count_prompt(body)  # type: ignore[union-attr]
        except Exception as e:
            self._degrade(e)
            return None
        if n is None:
            self._degrade(NotSupported("the provider cannot count prompt tokens"))
        return n

    def render(self, body: dict[str, Any]) -> str:
        if self.method == "server":
            try:
                return self.llm.apply_template(body)  # type: ignore[union-attr]
            except Exception as e:
                self._degrade(e)
        from argus.mock.server import render_chatml  # template approximation

        return render_chatml(body)

    def prompt(self, body: dict[str, Any]) -> int:
        """Tokens of a chat request as the model sees it (template applied)."""
        if self.method == "server":
            text = self.render(body)
            if self.method == "server":
                return self.count(text)
        if self.method == "api":
            n = self._api_prompt(body)
            if n is not None:
                return n
        return self.count(self.render(body))


SAMPLE_CALL = ("edit", {"path": "src/app.py", "old": "return a - b", "new": "return a + b"})


def call_cost(protocol: str, counter: TokenCounter) -> int:
    """Output tokens needed to express one representative tool call."""
    name, args = SAMPLE_CALL
    if protocol == "native":  # Hermes format used by Qwen chat templates
        text = f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n</tool_call>'
    elif protocol == "json_schema":  # llama-server's schema grammar allows spaces; models use them
        text = json.dumps({"tool": name, "args": args})
    else:  # argus GBNF: no insignificant whitespace
        text = json.dumps({"tool": name, "args": args}, separators=(",", ":"))
    return counter.count(text)


@dataclass
class Overhead:
    method: str  # server | estimate
    protocol: str
    total: int  # prompt tokens added before the task text
    system: int  # the system prompt text alone
    parts: dict[str, int]  # system prompt parts
    tools: int  # tokens added by the `tools` field (native: template preamble + schemas)
    per_tool: dict[str, int] = field(default_factory=dict)  # marginal cost of each tool
    framing: int = 0  # chat-template tokens around the system prompt
    call_cost: int = 0  # output tokens for a sample call
    constraint_chars: int = 0  # size of the schema/grammar sent (not in the prompt)
    context_window: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(agent: Agent, detailed: bool = True, counter: TokenCounter | None = None) -> Overhead:
    counter = counter or TokenCounter(agent.llm)
    proto = agent.protocol
    parts = agent.prompt_parts()
    system_text = parts.render()
    fields = proto.request_fields()
    tools = fields.get("tools") or []
    sys_msg = {"role": "system", "content": system_text}
    user = {"role": "user", "content": "x"}

    def P(messages: list[dict[str, Any]], tool_list: list[Any] | None = None) -> int:
        body: dict[str, Any] = {"messages": messages}
        if tool_list:
            body["tools"] = tool_list
        return counter.prompt(body)

    full = P([sys_msg, user], tools)
    bare = P([user])
    no_tools = P([sys_msg, user]) if tools else full
    exact = detailed  # with a request-counting API, exact part counts cost one call each
    part_counts = {
        "base": counter.count(parts.base, exact),
        "protocol": counter.count(parts.protocol, exact),
        "agents_md": counter.count(parts.agents_md, exact),
        "skills": counter.count(parts.skills, exact),
    }
    for i, extra in enumerate(parts.extra):
        part_counts[f"extra{i}"] = counter.count(extra, exact)
    system = counter.count(system_text, exact)
    ov = Overhead(
        method=counter.method,
        protocol=proto.name,
        total=full - bare,
        system=system,
        parts=part_counts,
        tools=full - no_tools,
    )
    ov.framing = ov.total - ov.system - ov.tools
    constraint = {k: v for k, v in fields.items() if k in ("grammar", "response_format")}
    if constraint:
        ov.constraint_chars = len(constraint.get("grammar") or json.dumps(constraint))
    if detailed:
        if tools:
            for i, t in enumerate(tools):
                rest = tools[:i] + tools[i + 1 :]
                ov.per_tool[t["function"]["name"]] = full - P([sys_msg, user], rest or None)
        else:
            for t in agent.tools.values():
                ov.per_tool[t.name] = counter.count(t.signature(), exact)
        ov.call_cost = call_cost(proto.name, counter)
        ov.context_window = (
            agent.context_window()
            if counter.method in ("server", "api")
            else agent.cfg.model.context_window
        )
    if counter.method == "api":
        ov.notes.append("prompt counts from the provider's token-counting API; parts by difference")
    elif counter.method != "server":
        ov.notes.append(
            f"estimated (~{CHARS_PER_TOKEN:.1f} chars/token, approximate template): "
            f"{counter.error or 'no server'}"
        )
    return ov


def format_overhead(rows: list[tuple[str, Overhead]]) -> str:
    """Side-by-side table for one or more measurements."""
    names = [n for n, _ in rows]
    ovs = [o for _, o in rows]
    width = max(12, *(len(n) for n in names))

    def line(label: str, values: list[Any]) -> str:
        return f"{label:<28}" + "".join(f"{v!s:>{width + 2}}" for v in values)

    out = [line("", names), line("method", [o.method for o in ovs])]
    out.append(line("TOTAL overhead (prompt tok)", [o.total for o in ovs]))
    out.append(line("  system prompt", [o.system for o in ovs]))
    for key in ("base", "protocol", "agents_md", "skills"):
        if any(o.parts.get(key) for o in ovs):
            out.append(line(f"    {key}", [o.parts.get(key, 0) for o in ovs]))
    out.append(line("  tools field", [o.tools for o in ovs]))
    tool_names: list[str] = []
    for o in ovs:
        tool_names += [t for t in o.per_tool if t not in tool_names]
    for t in tool_names:
        out.append(line(f"    {t}", [o.per_tool.get(t, "-") for o in ovs]))
    out.append(line("  template framing", [o.framing for o in ovs]))
    out.append(line("sample call (output tok)", [o.call_cost for o in ovs]))
    out.append(line("constraint size (chars)", [o.constraint_chars or "-" for o in ovs]))
    if any(o.context_window for o in ovs):
        out.append(
            line(
                "share of context window",
                [
                    f"{100 * o.total / o.context_window:.1f}%" if o.context_window else "-"
                    for o in ovs
                ],
            )
        )
    notes = sorted({n for o in ovs for n in o.notes})
    return "\n".join(out + [f"note: {n}" for n in notes])
