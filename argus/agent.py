"""The tool loop."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from argus.compact import HEAD, SUMMARY_PREFIX, Compaction, Compactor
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
from argus.fsstate import (
    Change,
    Checkpoint,
    Checkpointer,
    check_edit,
    mass_deletion,
    summarize,
)
from argus.llm import Completion, ContextOverflow, Delta, LLMClient, LLMError
from argus.prompt import PromptParts, base_prompt
from argus.protocols import Parsed, Protocol, ToolCall, make_protocol
from argus.skills import (
    Skill,
    SkillTool,
    discover_skills,
    load_agents_md,
    render_agents_md,
    render_skill_index,
    skill_for_path,
)
from argus.store import Store, new_id
from argus.tokens import TokenCounter, measure
from argus.tools import ToolContext, ToolError, ToolResult, build_tools
from argus.tools.base import Tool, digest

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
        self.agents_md: list[tuple[str, str]] = []
        self.skills: list[Skill] = []
        self.context_errors: list[str] = []
        self.load_context()
        if self.skills:
            self.tools["skill"] = SkillTool(self.skills, self.executor)
        self.protocol: Protocol = make_protocol(cfg.agent.protocol, self.tools, cfg.agent)
        self._n_ctx: int | None = None
        self.counter = TokenCounter(self.llm)
        self.compactor = Compactor(cfg.compaction, self.counter)

    def load_context(self) -> None:
        """AGENTS.md files and skills. Failures are recorded, never fatal."""
        c = self.cfg.context
        if c.agents_md:
            try:
                self.agents_md = load_agents_md(self.executor, c)
            except Exception as e:
                self.context_errors.append(f"AGENTS.md: {type(e).__name__}: {e}")
        if c.skills:
            try:
                self.skills = discover_skills(self.executor, c)
            except Exception as e:
                self.context_errors.append(f"skills: {type(e).__name__}: {e}")

    def prompt_parts(self) -> PromptParts:
        base = base_prompt(
            self.cfg.agent.system_prompt, self.executor.workdir, self.protocol.finish_hint()
        )
        return PromptParts(
            base=base,
            protocol=self.protocol.prompt_section(),
            agents_md=render_agents_md(self.agents_md, self.cfg.context.max_agents_md_chars),
            skills=render_skill_index(self.skills),
        )

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
        after_turn: Callable[[int, _Run], None] | None = None,
    ) -> RunResult:
        """Run one task. ``after_turn(turn, run)`` is called after each tool turn."""
        run = _Run(self, task, task_id=task_id, batch_id=batch_id, variant=variant)
        run.after_turn = after_turn
        return run.execute()

    def close(self) -> None:
        self.compactor.close()
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
        self.ckpt: Checkpointer | None = None
        self.after_turn: Callable[[int, _Run], None] | None = None
        self.turn_mutated = False  # a mutating tool ran this turn
        self.t0 = time.perf_counter()
        self.last_prompt: int | None = None  # prompt tokens of the latest request
        self.last_max_tokens: int | None = None
        self.last_context_ids: list[int] = []
        self.projection_from = 0  # context index from which messages are not yet counted

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
        try:
            if self.cfg.log.measure_overhead:
                self.record_overhead()
            if self.cfg.checkpoint.enabled:
                self.start_checkpoints()
            self.store.add_event(
                self.id,
                None,
                "context",
                {
                    "agents_md": [src for src, _ in a.agents_md],
                    "skills": [
                        {"name": s.name, "path": s.path, "local": s.local} for s in a.skills
                    ],
                    "errors": a.context_errors,
                },
            )
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
            if self.result.status == "running":  # every exit path should have set a status
                self.result.error = self.result.error or "run ended without a status"
                self.finish("error")
            self.final_checkpoint()
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
            self.turn_mutated = False
            if self.over_budget(turn):
                return
            if cfg.max_wall_seconds and time.perf_counter() - self.t0 > cfg.max_wall_seconds:
                self.fail(turn, "timeout", f"run exceeded {cfg.max_wall_seconds:g}s")
                self.finish("failed")
                return
            self.maybe_compact(turn)
            self.rep.turn_start(turn)
            got = self.generate_turn(turn)
            if got is None:
                return
            c, parsed = got
            self.add(parsed.assistant, turn)
            self.last_prompt = c.prompt_tokens
            self.projection_from = len(self.context) - 1  # the new assistant message onwards
            self.rep.turn_end(turn, c, parsed)

            if c.finish_reason == "length" and not parsed.calls:
                self.fail(turn, "token_cap", f"turn hit max_tokens ({self.last_max_tokens})")
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
            if self.after_turn:
                self.after_turn(turn, self)
                if self.result.status != "running":
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
                    f"reasoning used all {self.last_max_tokens} max_tokens without acting",
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

    def generate(
        self, turn: int, extra: dict[str, Any] | None = None, emergency: bool = False
    ) -> Completion | None:
        body = self.request(extra)
        self.last_context_ids = [mid for mid, _ in self.context]
        self.last_max_tokens = body.get("max_tokens")
        try:
            c = self.agent.llm.chat(body, stream=self.cfg.model.stream, monitors=self.monitors())
        except ContextOverflow as e:
            self.fail(turn, "token_cap", f"context overflow: {e}")
            if not emergency and self.compact(turn, force=True):
                self.rep.note("compacted after context overflow; retrying")
                return self.generate(turn, extra, emergency=True)
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
        # Streamed: one chunk per token. Otherwise count the text with the server's tokenizer.
        count = self.agent.counter.count
        reasoning_tokens = c.reasoning_chunks or (count(c.reasoning) if c.reasoning else 0)
        content_tokens = c.content_chunks or (count(c.content) if c.content else 0)
        r.turns = turn + 1
        r.prompt_tokens += c.prompt_tokens
        r.completion_tokens += c.completion_tokens
        r.reasoning_tokens += reasoning_tokens
        r.max_context = max(r.max_context, c.prompt_tokens + c.completion_tokens)
        t = c.timings
        self.store.add_turn(
            self.id,
            turn,
            attempt=attempt,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            cached_tokens=c.cached_tokens,
            reasoning_tokens=reasoning_tokens,
            content_tokens=content_tokens,
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
            max_tokens=self.last_max_tokens,
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
            pre, changes = None, None
            if call.error:
                self.fail(turn, "malformed_call", call.error)
                res, ms = ToolResult(f"Error: {call.error}", ok=False, error=call.error), 0.0
            else:
                valid += 1
                if call.salvaged:
                    self.fail(
                        turn, "malformed_call", f"salvaged {call.name} call from unparsed text"
                    )
                tool = self.agent.tools[call.name]
                self.turn_mutated |= tool.mutating
                pre = self.pre_checkpoint(turn, i, call) if tool.mutating else None
                res, ms = self.execute_tool(call)
                changes = self.post_check(turn, call, res, pre) if pre else None
                self.log_skill_use(turn, call, res)
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
                fs_changes_json=[[c.status, c.path] for c in changes] if changes else None,
                checkpoint=pre.commit if pre else None,
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

    def log_skill_use(self, turn: int, call: ToolCall, res: ToolResult) -> None:
        """Record skill loads: via the skill tool, or by reading/cat-ing a SKILL.md."""
        for ev in self.tctx.events:
            if ev.get("kind") == "skill":
                self.store.add_skill_invocation(self.id, turn, ev["skill"], ev["path"], ev["via"])
        self.tctx.events.clear()
        skills = self.agent.skills
        if not skills or not res.ok:
            return
        ex = self.agent.executor
        if call.name == "read":
            s = skill_for_path(skills, ex.resolve(call.args.get("path", "")))
            if s:
                self.store.add_skill_invocation(self.id, turn, s.name, s.path, "read")
        elif call.name == "bash" and "SKILL.md" in call.args.get("cmd", ""):
            cmd = call.args["cmd"]
            for s in skills:
                if not s.local and (s.path in cmd or ex.rel(s.path) in cmd):
                    self.store.add_skill_invocation(self.id, turn, s.name, s.path, "bash")

    # -- compaction ------------------------------------------------------------------------------

    def projected_tokens(self) -> int:
        """Prompt size of the next request: last prompt + messages added since."""
        comp = self.agent.compactor
        added = sum(comp.tokens(m) for _, m in self.context[self.projection_from :])
        return (self.last_prompt or 0) + added

    def maybe_compact(self, turn: int) -> None:
        if not self.cfg.compaction.enabled or self.last_prompt is None:
            return
        n_ctx = self.agent.context_window()
        if n_ctx and self.projected_tokens() > self.agent.compactor.limit(n_ctx):
            self.compact(turn)

    def compact(self, turn: int, force: bool = False) -> bool:
        """Shrink the context; returns True if anything changed."""
        cc = self.cfg.compaction
        if not cc.enabled:
            return False
        comp = self.agent.compactor
        n_ctx = self.agent.context_window()
        limit = comp.limit(n_ctx) if n_ctx else 0
        projected = self.projected_tokens()
        changed = False
        if cc.mask:
            t0 = time.perf_counter()
            replacements, saved = comp.mask(self.context)
            if replacements:
                for i, new in replacements:
                    mid = self.store.add_message(self.id, turn, new, kind="masked")
                    self.context[i] = (mid, new)
                self.log_compaction(
                    turn,
                    Compaction("mask", projected, projected - saved, len(replacements), _ms(t0)),
                )
                projected -= saved
                changed = True
            if not force and projected <= limit:
                return self.reset_projection(projected, changed)
        hi = comp.middle_end(self.context)
        if hi <= HEAD + 1:
            self.rep.note("nothing left to compact")
            return self.reset_projection(projected, changed)
        t0 = time.perf_counter()
        removed = self.context[HEAD:hi]
        record = Compaction("summarize", projected, 0, len(removed), 0.0, model=cc.model)
        summary = None
        if cc.summarize and comp.llm is not None:
            try:
                summary = comp.summarize(self.task, self.context, hi)
            except Exception as e:  # small model down: fall back to dropping
                record.error = f"{type(e).__name__}: {e}"
        if summary:
            msg = {"role": "user", "content": SUMMARY_PREFIX + summary}
            record.summary = summary
        else:
            record.stage, record.model = "drop", ""
            msg = {
                "role": "user",
                "content": f"[{len(removed)} earlier messages were removed to save context]",
            }
        saved = sum(comp.tokens(m) for _, m in removed) - comp.tokens(msg)
        mid = self.store.add_message(self.id, turn, msg, kind=record.stage)
        self.context = self.context[:HEAD] + [(mid, msg)] + self.context[hi:]
        record.tokens_after = projected - saved
        record.ms = _ms(t0)
        self.log_compaction(turn, record)
        return self.reset_projection(record.tokens_after, True)

    def reset_projection(self, projected: int, changed: bool) -> bool:
        if changed:
            self.last_prompt = projected
            self.projection_from = len(self.context)
        return changed

    def log_compaction(self, turn: int, c: Compaction) -> None:
        self.store.add_compaction(
            self.id,
            turn,
            stage=c.stage,
            tokens_before=c.tokens_before,
            tokens_after=c.tokens_after,
            messages_affected=c.affected,
            duration_ms=round(c.ms, 1),
            model=c.model or None,
            summary=c.summary or None,
        )
        if c.error:
            self.store.add_event(self.id, turn, "compaction_error", c.error)
        self.rep.note(f"compacted ({c.stage}): ~{c.tokens_before} → ~{c.tokens_after} tokens")

    # -- checkpoints and filesystem checks -------------------------------------------------------

    def start_checkpoints(self) -> None:
        self.ckpt = Checkpointer(self.agent.executor, self.cfg.checkpoint, self.id)
        cp = self.ckpt.start()
        if cp is None:
            self.store.add_event(self.id, None, "checkpoints_disabled", self.ckpt.error)
            self.rep.note(f"checkpoints disabled: {self.ckpt.error}")
            return
        self.store.add_checkpoint(self.id, None, None, cp.commit, cp.tree, "baseline")

    def disable_checkpoints(self, err: Exception) -> None:
        if self.ckpt:
            self.ckpt.error = str(err)
        self.store.add_event(self.id, self.turn, "checkpoints_disabled", str(err))
        self.rep.note(f"checkpoints disabled: {err}")

    def pre_checkpoint(self, turn: int, i: int, call: ToolCall) -> Checkpoint | None:
        if not (self.ckpt and self.ckpt.active):
            return None
        try:
            cp = self.ckpt.checkpoint(
                f"turn {turn}: before {call.name}({brief_args(call.args, 60)})"
            )
        except Exception as e:
            self.disable_checkpoints(e)
            return None
        self.store.add_checkpoint(self.id, turn, i, cp.commit, cp.tree, f"before {call.name}")
        return cp

    def post_check(
        self, turn: int, call: ToolCall, res: ToolResult, pre: Checkpoint
    ) -> list[Change] | None:
        """Compare the workspace with the pre-call checkpoint and check the effect."""
        ex = self.agent.executor
        try:
            _, changes = self.ckpt.changes_since(pre)
        except Exception as e:
            self.disable_checkpoints(e)
            return None
        if call.name == "edit" and res.ok:
            path = res.meta.get("path", "")
            rel = ex.rel(path)
            if not changes and self.ckpt.git and self.ckpt.git.is_ignored(rel):
                res.meta["untracked"] = True  # ignored paths are invisible to snapshots
            else:
                problem = check_edit(changes, rel)
                if problem:
                    self.fail(turn, "fs_violation", problem)
            try:
                on_disk = digest(ex.read_bytes(path))
            except OSError as e:
                on_disk = f"unreadable: {e}"
            if res.meta.get("sha1") and on_disk != res.meta["sha1"]:
                self.fail(turn, "fs_violation", f"{rel} on disk differs from what edit wrote")
        elif changes and not res.ok:
            self.fail(
                turn, "fs_violation", f"failed {call.name} still changed: {summarize(changes)}"
            )
        if changes and call.name != "edit":
            res.text += f"\n[files changed: {summarize(changes)}]"
            for c in changes:
                self.tctx.tracker.invalidate(ex.resolve(c.path))
            problem = mass_deletion(changes)
            if problem:
                self.fail(turn, "fs_violation", problem)
        return changes

    def final_checkpoint(self) -> None:
        if not (self.ckpt and self.ckpt.active):
            return
        try:
            cp = self.ckpt.checkpoint("final")
            self.store.add_checkpoint(self.id, self.turn, None, cp.commit, cp.tree, "final")
        except Exception as e:
            self.store.add_event(self.id, self.turn, "checkpoint_error", str(e))

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


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)
