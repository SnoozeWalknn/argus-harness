"""The tool loop."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from argus.config import Config
from argus.detect import (
    LoopDetector,
    LoopVerdict,
    ReasoningBudget,
    RepetitionMonitor,
    claims_completion,
    cut_at_role_marker,
    periodic_tail,
    repeated_paragraph,
)
from argus.executors import Executor, make_executor
from argus.llm import Completion, ContextOverflow, Delta, LLMClient, LLMError
from argus.prompt import PromptParts, base_prompt
from argus.protocols import Parsed, Protocol, ToolCall, make_protocol
from argus.store import Store, new_id
from argus.tokens import TokenCounter, measure
from argus.tools import ToolContext, ToolError, ToolResult, build_tools
from argus.tools.base import Tool

CUT_OFF = (
    "Your reply was cut off at the output token limit. Be brief: make one tool call, "
    "or give the final answer."
)
RETRY_NUDGE = "Your previous reply ran away and was cut off. Act now: make the next tool call."


class Reporter:
    """Progress callbacks. The base class is silent; see ``argus.console`` for output."""

    def run_start(self, run_id: str, task: str) -> None: ...
    def turn_start(self, turn: int) -> None: ...
    def delta(self, d: Delta) -> None: ...
    def turn_end(self, turn: int, c: Completion, p: Parsed) -> None: ...
    def tool_start(self, turn: int, call: ToolCall) -> None: ...
    def tool_end(self, turn: int, call: ToolCall, result: ToolResult, ms: float) -> None: ...
    def failure(self, turn: int | None, tag: str, detail: str) -> None: ...
    def note(self, text: str) -> None: ...
    def run_end(self, result: RunResult) -> None: ...


@dataclass
class RunResult:
    run_id: str
    status: str = "running"  # completed | failed | error | interrupted
    final: str = ""
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    max_context: int = 0
    wall_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def tags(self) -> list[str]:
        return sorted({t for t, _ in self.failures})


def brief_args(args: dict[str, Any], limit: int = 80) -> str:
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v)
        s = s.replace("\n", "⏎")
        parts.append(f"{k}={s[:40]}{'…' if len(s) > 40 else ''}")
    out = ", ".join(parts)
    return out if len(out) <= limit else out[: limit - 1] + "…"


class Agent:
    """Owns long-lived resources (client, executor, store); each :meth:`run` is independent."""

    def __init__(
        self,
        cfg: Config,
        *,
        executor: Executor | None = None,
        llm: LLMClient | None = None,
        store: Store | None = None,
        reporter: Reporter | None = None,
    ):
        self.cfg = cfg
        self.executor = executor or make_executor(cfg.executor, cfg.tools.capture_bytes)
        m = cfg.model
        self.llm = llm or LLMClient(m.base_url, m.api_key, m.timeout, m.connect_timeout, m.retries)
        self.store = store or Store(cfg.db_path())
        self.reporter = reporter or Reporter()
        self.tools: dict[str, Tool] = build_tools(cfg.tools)
        self.protocol: Protocol = make_protocol(cfg.agent.protocol, self.tools, cfg.agent)
        self._n_ctx: int | None = None
        self.counter = TokenCounter(self.llm)

    def prompt_parts(self) -> PromptParts:
        base = base_prompt(
            self.cfg.agent.system_prompt, self.executor.workdir, self.protocol.finish_hint()
        )
        return PromptParts(base=base, protocol=self.protocol.prompt_section())

    def context_window(self) -> int:
        if self._n_ctx is None:
            self._n_ctx = self.cfg.model.context_window or self.llm.context_window()
        return self._n_ctx

    def sampling(self) -> dict[str, Any]:
        m = self.cfg.model
        body: dict[str, Any] = {"model": m.model}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "seed",
        ):
            value = getattr(m, key)
            if value is not None:
                body[key] = value
        if m.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": m.enable_thinking}
        return body

    def run(
        self,
        task: str,
        *,
        task_id: str | None = None,
        batch_id: str | None = None,
        variant: str | None = None,
    ) -> RunResult:
        return _Run(self, task, task_id=task_id, batch_id=batch_id, variant=variant).execute()

    def close(self) -> None:
        self.llm.close()
        self.executor.close()


class _Run:
    """State of a single run."""

    def __init__(self, agent: Agent, task: str, **meta: Any):
        self.agent = agent
        self.cfg = agent.cfg
        self.store = agent.store
        self.rep = agent.reporter
        self.protocol = agent.protocol
        self.task = task
        self.meta = meta
        self.id = new_id()
        self.result = RunResult(self.id)
        self.tctx = ToolContext(agent.executor, self.cfg.tools)
        self.context: list[tuple[int, dict[str, Any]]] = []  # (message id, message)
        self.turn = 0
        a = self.cfg.agent
        self.loops = LoopDetector(a.loop_repeat, a.loop_abort)
        self.loop_tagged: set[str] = set()
        self.idle_turns = 0  # consecutive turns without a valid tool call
        self.claimed_done: int | None = None  # turn where the model declared completion
        self.overrun_tagged = False

    # -- bookkeeping -----------------------------------------------------------------------

    def add(self, msg: dict[str, Any], turn: int | None, kind: str = "normal") -> None:
        mid = self.store.add_message(self.id, turn, msg, kind)
        self.context.append((mid, msg))

    def fail(self, turn: int | None, tag: str, detail: str = "") -> None:
        self.store.add_failure(self.id, turn, tag, detail)
        self.result.failures.append((tag, detail))
        self.rep.failure(turn, tag, detail)

    def finish(self, status: str, final: str = "") -> None:
        self.result.status = status
        self.result.final = final

    # -- main ----------------------------------------------------------------------------------

    def execute(self) -> RunResult:
        a = self.agent
        t0 = time.perf_counter()
        parts = a.prompt_parts()
        self.store.start_run(
            self.id,
            task=self.task,
            task_id=self.meta.get("task_id"),
            batch_id=self.meta.get("batch_id"),
            variant=self.meta.get("variant"),
            config_name=self.cfg.name,
            config_hash=self.cfg.hash(),
            config_json=json.dumps(self.cfg.to_dict(), default=str),
            workspace=a.executor.workdir,
            executor=a.executor.describe(),
            protocol=self.protocol.name,
            model=self.cfg.model.model,
        )
        self.rep.run_start(self.id, self.task)
        if self.cfg.log.measure_overhead:
            self.record_overhead()
        try:
            self.add({"role": "system", "content": parts.render()}, None)
            self.add({"role": "user", "content": self.task}, None)
            self.loop()
        except KeyboardInterrupt:
            self.fail(self.turn, "interrupted", "KeyboardInterrupt")
            self.finish("interrupted")
        except Exception as e:  # keep the partial log; never lose a run to a harness bug
            self.result.error = f"{type(e).__name__}: {e}"
            self.store.add_event(self.id, self.turn, "exception", traceback.format_exc())
            self.finish("error")
        finally:
            r = self.result
            r.wall_ms = (time.perf_counter() - t0) * 1000
            self.store.update_run(
                self.id,
                status=r.status,
                final_answer=r.final,
                n_turns=r.turns,
                n_tool_calls=r.tool_calls,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                reasoning_tokens=r.reasoning_tokens,
                max_context_tokens=r.max_context,
                wall_ms=round(r.wall_ms, 1),
                llm_ms=round(r.llm_ms, 1),
                tool_ms=round(r.tool_ms, 1),
                error=r.error,
                ended_at=time.time(),
            )
            self.rep.run_end(r)
        return self.result

    def record_overhead(self) -> None:
        a = self.agent
        try:
            ov = measure(a, detailed=False, counter=a.counter)
            self.store.update_run(
                self.id,
                overhead_tokens=ov.total,
                overhead_json=json.dumps(ov.to_dict()),
                context_window=a.context_window() or None,
            )
        except Exception as e:  # measurement must never break a run
            self.store.add_event(self.id, None, "overhead_error", f"{type(e).__name__}: {e}")

    def loop(self) -> None:
        cfg = self.cfg.agent
        for turn in range(cfg.max_turns):
            self.turn = turn
            if self.over_budget(turn):
                return
            self.rep.turn_start(turn)
            got = self.generate_turn(turn)
            if got is None:
                return
            c, parsed = got
            self.add(parsed.assistant, turn)
            self.rep.turn_end(turn, c, parsed)

            if c.finish_reason == "length" and not parsed.calls:
                self.fail(turn, "token_cap", f"turn hit max_tokens ({self.max_tokens()})")
                self.check_degenerate(turn, c)
                self.feedback(turn, CUT_OFF)
                if self.no_progress(turn, "token_cap"):
                    return
                continue
            if parsed.final is not None:
                self.finish("completed", self.check_final(turn, c, parsed.final))
                return
            self.check_claims(turn, parsed)
            valid = self.run_calls(turn, parsed)
            if self.result.status != "running":
                return
            for problem in parsed.problems:
                self.fail(turn, "malformed_call", problem)
                self.feedback(turn, f"Error: {problem}")
            if valid:
                self.idle_turns = 0
            elif self.no_progress(turn, "malformed_call"):
                return
        self.fail(cfg.max_turns - 1, "max_turns", f"no final answer after {cfg.max_turns} turns")
        self.finish("failed")

    def generate_turn(self, turn: int) -> tuple[Completion, Parsed] | None:
        """Generate one turn, retrying once if the generation ran away."""
        extra: dict[str, Any] | None = None
        for attempt in range(2):
            c = self.generate(turn, extra)
            if c is None:
                return None
            parsed = self.protocol.parse(c, turn)
            self.record_turn(turn, c, parsed, attempt)
            thinking_only = bool(c.reasoning.strip()) and not c.content.strip() and not parsed.calls
            if c.aborted == "repetition":
                self.fail(turn, "loop", c.abort_detail)
            elif c.aborted:
                self.fail(turn, "token_cap", c.abort_detail or c.aborted)
            elif c.finish_reason == "length" and thinking_only:
                self.fail(
                    turn,
                    "token_cap",
                    f"reasoning used all {self.max_tokens()} max_tokens without acting",
                )
                self.check_degenerate(turn, c)
            else:
                return c, parsed
            if attempt == 0 and self.cfg.agent.reasoning_retry:
                # Retry the turn, without thinking if the reasoning was what ran away.
                extra = self.protocol.disable_thinking() if thinking_only else None
                self.rep.note("retrying turn" + (" with thinking disabled" if extra else ""))
                self.feedback(turn, RETRY_NUDGE)
                continue
            break
        self.finish("failed")
        return None

    def no_progress(self, turn: int, tag: str) -> bool:
        """Count a turn without a valid call; abort after agent.max_malformed in a row."""
        self.idle_turns += 1
        if self.idle_turns >= self.cfg.agent.max_malformed:
            self.fail(
                turn,
                tag,
                f"aborting after {self.idle_turns} consecutive turns without a valid tool call",
            )
            self.finish("failed")
            return True
        return False

    def check_degenerate(self, turn: int, c: Completion) -> None:
        window = self.cfg.agent.repetition_window
        for kind, text in (("reasoning", c.reasoning), ("content", c.content)):
            unit = periodic_tail(text, window)
            if unit:
                self.fail(turn, "loop", f"{kind} degenerated into repeating {unit[:60]!r}")
                return

    def check_final(self, turn: int, c: Completion, final: str) -> str:
        """Tag and trim output that runs past the final answer."""
        clean, marker = cut_at_role_marker(final)
        if marker:
            self.fail(turn, "overrun", f"generation continued past the final answer ({marker})")
        para = repeated_paragraph(clean)
        if para:
            self.fail(turn, "overrun", f"final answer repeats itself: {para[:80]!r}")
        return clean.strip()

    def check_claims(self, turn: int, parsed: Parsed) -> None:
        """Tag tool turns that continue well after the model said the task was complete."""
        if self.claimed_done is None and claims_completion(parsed.content):
            self.claimed_done = turn
            return
        n = self.cfg.agent.overrun_turns
        if (
            self.claimed_done is not None
            and not self.overrun_tagged
            and turn - self.claimed_done >= n
        ):
            self.overrun_tagged = True
            self.fail(
                turn,
                "overrun",
                f"still calling tools {turn - self.claimed_done} turns after declaring completion "
                f"at turn {self.claimed_done}",
            )

    # -- model ---------------------------------------------------------------------------------

    def max_tokens(self) -> int:
        n = self.cfg.model.max_tokens
        budget = self.cfg.agent.token_budget
        if budget:
            n = max(1, min(n, budget - self.result.completion_tokens))
        return n

    def request(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Protocol fields, then sampling config, then extra_body, then per-call extras."""
        body = merge(self.protocol.request_fields(), self.agent.sampling())
        body["messages"] = [m for _, m in self.context]
        body["max_tokens"] = self.max_tokens()
        return merge(merge(body, self.cfg.model.extra_body), extra or {})

    def generate(self, turn: int, extra: dict[str, Any] | None = None) -> Completion | None:
        body = self.request(extra)
        self.last_context_ids = [mid for mid, _ in self.context]
        try:
            c = self.agent.llm.chat(body, stream=self.cfg.model.stream, monitors=self.monitors())
        except ContextOverflow as e:
            self.fail(turn, "token_cap", f"context overflow: {e}")
            self.finish("failed")
            return None
        except LLMError as e:
            self.fail(turn, "server_error", str(e))
            self.result.error = str(e)
            self.finish("error")
            return None
        self.result.llm_ms += c.total_ms
        return c

    def monitors(self) -> list:
        a = self.cfg.agent
        ms: list = [self.rep.delta]
        if a.max_reasoning_tokens:
            ms.append(ReasoningBudget(a.max_reasoning_tokens))
        if a.repetition_window:
            ms.append(RepetitionMonitor(a.repetition_window))
        return ms

    def record_turn(self, turn: int, c: Completion, p: Parsed, attempt: int = 0) -> None:
        r = self.result
        r.turns = turn + 1
        r.prompt_tokens += c.prompt_tokens
        r.completion_tokens += c.completion_tokens
        r.reasoning_tokens += c.reasoning_chunks
        r.max_context = max(r.max_context, c.prompt_tokens + c.completion_tokens)
        t = c.timings
        self.store.add_turn(
            self.id,
            turn,
            attempt=attempt,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            cached_tokens=c.cached_tokens,
            reasoning_tokens=c.reasoning_chunks,
            content_tokens=c.content_chunks,
            reasoning=c.reasoning,
            content=c.content,
            tool_calls_json=[{"id": x.id, "name": x.name, "arguments": x.raw} for x in p.calls],
            finish_reason=c.finish_reason,
            aborted=c.aborted,
            ttft_ms=_r(c.ttft_ms),
            prompt_ms=_r(t.get("prompt_ms")),
            gen_ms=_r(t.get("predicted_ms")),
            total_ms=_r(c.total_ms),
            prompt_tps=_r(t.get("prompt_per_second")),
            gen_tps=_r(t.get("predicted_per_second")),
            max_tokens=self.max_tokens(),
            context_ids=self.last_context_ids,
            raw_response=c.raw,
        )

    def feedback(self, turn: int, text: str) -> None:
        self.add(self.protocol.feedback_message(text), turn, kind="feedback")

    def over_budget(self, turn: int) -> bool:
        budget = self.cfg.agent.token_budget
        if budget and self.result.completion_tokens >= budget:
            self.fail(turn, "token_cap", f"run token budget exhausted ({budget})")
            self.finish("failed")
            return True
        return False

    # -- tools ---------------------------------------------------------------------------------

    def run_calls(self, turn: int, parsed: Parsed) -> int:
        """Execute the turn's calls; returns how many were valid."""
        valid = 0
        for i, call in enumerate(parsed.calls):
            self.rep.tool_start(turn, call)
            if call.error:
                self.fail(turn, "malformed_call", call.error)
                res, ms = ToolResult(f"Error: {call.error}", ok=False, error=call.error), 0.0
            else:
                valid += 1
                if call.salvaged:
                    self.fail(
                        turn, "malformed_call", f"salvaged {call.name} call from unparsed text"
                    )
                res, ms = self.execute_tool(call)
                verdict = self.loops.observe(
                    call.name, call.args, res.text, f"{call.name}({brief_args(call.args, 60)})"
                )
                if verdict:
                    self.on_loop(turn, verdict, res)
            self.rep.tool_end(turn, call, res, ms)
            self.result.tool_calls += 1
            self.result.tool_ms += ms
            self.add(self.protocol.result_message(call, res.text), turn)
            self.store.add_tool_call(
                self.id,
                turn,
                i,
                call_id=call.id,
                name=call.name,
                args_json=call.args,
                raw_args=call.raw,
                ok=int(res.ok),
                error=res.error,
                result=res.text,
                result_chars=len(res.text),
                truncated=int(res.truncated),
                duration_ms=round(ms, 1),
                meta_json=res.meta or None,
            )
            if self.result.status != "running":
                break
        return valid

    def on_loop(self, turn: int, verdict: LoopVerdict, res: ToolResult) -> None:
        if verdict.abort:
            self.fail(turn, "loop", f"{verdict.detail}; aborting")
            self.finish("failed")
            return
        if verdict.key not in self.loop_tagged:
            self.loop_tagged.add(verdict.key)
            self.fail(turn, "loop", verdict.detail)
        res.text += f"\n[argus: {verdict.detail}. Nothing changed; try a different approach.]"

    def execute_tool(self, call: ToolCall) -> tuple[ToolResult, float]:
        tool = self.agent.tools[call.name]
        t0 = time.perf_counter()
        try:
            res = tool.run(self.tctx, call.args)
        except ToolError as e:
            res = ToolResult(f"Error: {e}", ok=False, error=str(e))
        except Exception as e:
            self.store.add_event(self.id, self.turn, "tool_exception", traceback.format_exc())
            res = ToolResult(f"Error: {type(e).__name__}: {e}", ok=False, error=f"internal: {e}")
        return res, (time.perf_counter() - t0) * 1000


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge that also merges nested dicts one level (e.g. chat_template_kwargs)."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)
