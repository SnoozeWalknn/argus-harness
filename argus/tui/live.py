"""What the model is doing right now: the token meter, and readable streaming text.

Pure functions and a small state object, so the TUI stays a thin view over them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def fmt_k(n: float) -> str:
    """1234 → 1.2k, 65536 → 66k."""
    if n < 1000:
        return f"{n:.0f}"
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class Meter:
    """Token output of the current turn, updated from streamed deltas."""

    phase: str = "idle"  # idle | waiting | thinking | writing | calling <tool> | running <tool>
    who: str = ""  # subagent name, "" for the main agent
    turn: int = 0
    tokens: int = 0  # generated this turn (estimated while streaming, exact at the end)
    run_started: float | None = None
    turn_started: float | None = None
    first_token: float | None = None
    last_tokens: int = 0  # the last finished turn
    last_rate: float = 0.0
    session_in: int = 0
    session_out: int = 0
    cost: float = 0.0
    context: int = 0  # tokens in the model's context after the last turn
    window: int = 0
    frame: int = 0
    history: list[float] = field(default_factory=list)  # tok/s of recent turns

    def start_run(self) -> None:
        self.run_started = time.monotonic()
        self.phase = "waiting"

    def start_turn(self, turn: int, who: str = "") -> None:
        self.turn, self.who = turn, who
        self.tokens = 0
        self.turn_started = time.monotonic()
        self.first_token = None
        self.phase = "waiting"

    def stream(self, kind: str, tokens: int, who: str = "") -> None:
        now = time.monotonic()
        if self.first_token is None:
            self.first_token = now
        self.who = who
        self.tokens = max(self.tokens, tokens)
        self.phase = {"reasoning": "thinking", "content": "writing"}.get(kind, "calling a tool")

    def end_turn(self, prompt: int, generated: int, who: str = "") -> None:
        self.tokens = generated or self.tokens
        self.last_tokens = self.tokens
        self.last_rate = self.rate()
        if self.last_rate:
            self.history = (self.history + [self.last_rate])[-12:]
        if not who:
            self.session_in += prompt
            self.session_out += generated
            self.context = prompt + generated
        self.phase = "waiting"

    def tool(self, name: str, who: str = "") -> None:
        self.who = who
        self.phase = f"running {name}"

    def stop(self) -> None:
        self.phase = "idle"
        self.who = ""
        self.run_started = None

    def rate(self) -> float:
        if self.first_token is None or not self.tokens:
            return 0.0
        dt = time.monotonic() - self.first_token
        return self.tokens / dt if dt > 0.05 else 0.0

    def running(self) -> bool:
        return self.phase != "idle"

    def lines(self) -> list[tuple[str, str]]:
        """(text, style) lines for the box, each at most ~26 columns; styles are
        theme-agnostic names the view maps to colours."""
        out: list[tuple[str, str]] = []
        if self.running():
            self.frame = (self.frame + 1) % len(SPINNER)
            who = f"{self.who}: " if self.who else ""
            out.append((f"{SPINNER[self.frame]} {who}{self.phase}"[:28], "phase"))
            rate = self.rate()
            out.append(
                (f"↓ {fmt_k(self.tokens)} tok" + (f" · {rate:.0f}/s" if rate else ""), "big")
            )
            elapsed = time.monotonic() - (self.run_started or time.monotonic())
            out.append((f"turn {self.turn + 1} · {clock(elapsed)}", "dim"))
        else:
            out.append(("○ idle", "phase-idle"))
            if self.last_tokens:
                rate = f" · {self.last_rate:.0f}/s" if self.last_rate else ""
                out.append((f"last ↓ {fmt_k(self.last_tokens)} tok{rate}", "dim"))
        if self.window and self.context:
            pct = 100 * self.context / self.window
            out.append((f"ctx {pct:.0f}% of {fmt_k(self.window)}", "warn" if pct > 80 else "dim"))
        if self.session_in or self.session_out:
            out.append((f"in {fmt_k(self.session_in)} · out {fmt_k(self.session_out)}", "dim"))
        if self.cost:
            out.append((f"${self.cost:.4f}", "dim"))
        if len(self.history) > 1:
            out.append((f"tok/s {sparkline(self.history)}", "spark"))
        return out


def sparkline(values: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return bars[4] * len(values)
    return "".join(bars[round((v - lo) / (hi - lo) * (len(bars) - 1))] for v in values)


def _json_string(raw: str, start: int) -> str:
    """Decode a JSON string body from ``start`` up to its closing quote or the end."""
    out, i = [], start
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "", "b": "", "f": ""}
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= len(raw):
                break
            nxt = raw[i + 1]
            if nxt == "u":
                code = raw[i + 2 : i + 6]
                if len(code) < 4:
                    break
                try:
                    out.append(chr(int(code, 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append(esc.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def constrained_view(raw: str) -> tuple[str, str]:
    """(thinking, saying) of a partial constrained-protocol reply: the text of a leading
    ``<think>`` block, and the JSON action's ``thought`` so far (or its tool name)."""
    think = ""
    rest = raw
    stripped = raw.lstrip()
    if stripped.startswith("<think>"):
        body = stripped[len("<think>") :]
        think, sep, rest = body.partition("</think>")
        if not sep:
            return think.strip(), ""
    key = rest.find('"thought"')
    if key != -1:
        colon = rest.find(":", key + 9)
        quote = rest.find('"', colon + 1) if colon != -1 else -1
        if quote != -1:
            return think.strip(), _json_string(rest, quote + 1)
    tool = rest.find('"tool"')
    if tool != -1:
        colon = rest.find(":", tool + 6)
        quote = rest.find('"', colon + 1) if colon != -1 else -1
        if quote != -1:
            name = _json_string(rest, quote + 1)
            if name:
                return think.strip(), f"→ {name}(…)"
    return think.strip(), ""
