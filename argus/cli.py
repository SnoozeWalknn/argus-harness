"""argus command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from argus.config import Config, ConfigError, load_config
from argus.fsstate import summarize
from argus.store import Store
from argus.suite import SuiteError

# argus run: 0 completed, 3 the model declined (the session stays open), 1 anything else
EXIT_CODES = {"completed": 0, "refused": 3}


def _config(args: argparse.Namespace, pick: bool = False) -> Config:
    """Config from the layered files, -m/-o and flags. ``pick``: choose a model when none
    is configured (a local server, else a cloud API with a key)."""
    from argus.defaults import config_layers, pick_model

    overrides = list(getattr(args, "override", None) or [])
    if getattr(args, "db", None):
        overrides.append(f"log.db={json.dumps(args.db)}")
    workdir = getattr(args, "workdir", None)
    if workdir:
        overrides.append(f"executor.workdir={json.dumps(workdir)}")
    spec = getattr(args, "model", None)
    if spec:
        overrides[:0] = model_overrides(spec)  # explicit -o still wins
    if getattr(args, "approval", None):
        overrides.insert(0, f"agent.approval={json.dumps(args.approval)}")
    if getattr(args, "plan", False):
        overrides.insert(0, 'agent.mode="plan"')
    if getattr(args, "record", None):
        overrides.append(f"model.record_dir={json.dumps(args.record)}")
    local_dir = workdir or os.getcwd()
    layers = config_layers(local_dir, getattr(args, "config", None))
    cfg = load_config(layers, overrides)
    if pick:
        chosen, why = pick_model(cfg.explicit)
        if chosen:
            cfg = load_config(layers, chosen + overrides)
            if not getattr(args, "quiet", False):
                print(f"argus: {why}", file=sys.stderr)
    if workdir and cfg.executor.kind == "local":  # an SSH workdir is a path on the remote host
        cfg.executor.workdir = str(Path(workdir).resolve())
    return cfg


def model_overrides(spec: str) -> list[str]:
    """``-m anthropic/claude-opus-5`` → provider and model overrides."""
    from argus.providers import parse_spec

    provider, model = parse_spec(spec)
    out = []
    if provider:
        out.append(f"model.provider={json.dumps(provider)}")
    if model:
        out.append(f"model.model={json.dumps(model)}")
    return out


def _store(args: argparse.Namespace) -> Store:
    if getattr(args, "db", None):
        return Store(args.db)
    cfg = load_config(getattr(args, "config", None)) if getattr(args, "config", None) else Config()
    return Store(cfg.db_path())


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="TOML config file")
    p.add_argument(
        "-m",
        "--model",
        metavar="SPEC",
        help="model as provider/model, e.g. anthropic/claude-opus-5, ollama/qwen3-coder:30b",
    )
    p.add_argument(
        "-o",
        "--override",
        action="append",
        metavar="KEY=VALUE",
        help="override a config key (repeatable)",
    )
    p.add_argument(
        "--db", help="SQLite log path (default: $ARGUS_DB or ~/.local/share/argus/argus.db)"
    )


def cmd_run(args: argparse.Namespace) -> int:
    from argus.agent import Agent
    from argus.console import ConsoleReporter

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text()
    elif task == "-":
        task = sys.stdin.read()
    if not task or not task.strip():
        print("argus: empty task", file=sys.stderr)
        return 2
    cfg = _config(args, pick=True)
    session = None
    if getattr(args, "cont", False) or getattr(args, "session", None):
        store = Store(cfg.db_path())
        try:
            workspace = cfg.executor.workdir or os.getcwd()
            session = store.resolve_session(
                args.session or "last", None if args.session else workspace
            )
        except KeyError as e:
            print(f"argus: {e.args[0]}", file=sys.stderr)
            return 2
        finally:
            store.close()
    reporter = ConsoleReporter(verbose=args.verbose) if not args.quiet else None
    agent = Agent(cfg, reporter=reporter, approver=_approver(args, cfg))
    try:
        result = agent.run(task, session=session)
    finally:
        agent.close()
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "session_id": agent.store.run(result.run_id)["session_id"],
                    "status": result.status,
                    "final": result.final,
                    "turns": result.turns,
                    "tool_calls": result.tool_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "failures": result.failures,
                    "error": result.error,
                },
                indent=2,
            )
        )
    elif result.final:
        print(result.final)
    return EXIT_CODES.get(result.status, 1)


def _approver(args: argparse.Namespace, cfg: Config):
    """Ask on the terminal when someone is there; otherwise agent.headless_approval decides."""
    from argus.approval import FixedApprover, TTYApprover

    if getattr(args, "yes", False):
        return FixedApprover("yes")
    if cfg.agent.approval in ("ask", "auto") and sys.stdin.isatty() and sys.stderr.isatty():
        return TTYApprover()
    return None


def cmd_init(args: argparse.Namespace) -> int:
    from argus.defaults import init

    for line in init(force=args.force):
        print(line)
    print("next: argus doctor, then argus (TUI) or argus run 'your task'")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from argus.defaults import doctor, format_doctor

    report = doctor()
    print(json.dumps(report, indent=2) if args.json else format_doctor(report))
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    from argus.tui import available

    if not available():
        print(
            "argus: the TUI needs Textual: pip install 'argus-harness[tui]' "
            "(or uv tool install 'argus-harness[tui]')",
            file=sys.stderr,
        )
        return 2
    from argus.tui.app import ArgusApp

    def make_config(spec: str | None) -> Config:
        a = argparse.Namespace(**vars(args))
        a.quiet = True
        if spec:
            a.model = spec
        return _config(a, pick=not spec and not args.model)

    session = None
    if args.session or args.cont:
        cfg = make_config(None)
        store = Store(cfg.db_path())
        try:
            workspace = cfg.executor.workdir or os.getcwd()
            session = store.resolve_session(
                args.session or "last", None if args.session else workspace
            )
        except KeyError as e:
            print(f"argus: {e.args[0]}", file=sys.stderr)
            return 2
        finally:
            store.close()
    ArgusApp(make_config, session=session).run()
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    store = _store(args)
    rows = store.sessions(limit=args.limit)
    if not rows:
        print("no sessions yet")
        return 0
    print(f"{'session':<24}{'runs':>5}  {'updated':<17}{'model':<28}title")
    for r in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"]))
        model = (r["model"] or "-")[:27]
        print(f"{r['id']:<24}{r['n_runs']:>5}  {when:<17}{model:<28}{r['title'] or ''}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    args.session = args.session_id or None
    args.cont = not args.session
    return cmd_run(args)


def cmd_runs(args: argparse.Namespace) -> int:
    store = _store(args)
    rows = store.runs(limit=args.limit)
    for r in rows:
        fails = {f["tag"] for f in store.failures(r["id"])}
        print(
            f"{r['id']}  {r['status']:<11} {r['protocol'] or '':<11} {r['n_turns'] or 0:>3}t "
            f"{r['completion_tokens'] or 0:>6}gen  {','.join(sorted(fails)) or '-':<24} "
            f"{(r['task'] or '').strip().splitlines()[0][:60] if r['task'] else ''}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from argus.console import render_run

    store = _store(args)
    try:
        run_id = store.resolve_run(args.run)
    except KeyError as e:
        print(f"argus: {e.args[0]}", file=sys.stderr)
        return 1
    print(render_run(store, run_id, full=args.full))
    return 0


def cmd_overhead(args: argparse.Namespace) -> int:
    from argus.agent import Agent
    from argus.config import PROTOCOLS, apply_override
    from argus.providers import provider_from_config
    from argus.tokens import TokenCounter, format_overhead, measure

    cfg = _config(args)
    protocols = PROTOCOLS if args.all else [cfg.agent.protocol]
    cfg.model.retries = 0
    llm = None if args.offline else provider_from_config(cfg.model)
    counter = TokenCounter(llm)  # shared, so every protocol is measured the same way
    rows = []
    for proto in protocols:
        apply_override(cfg, f"agent.protocol={json.dumps(proto)}")
        agent = Agent(cfg, store=Store(":memory:"), llm=llm)
        rows.append((proto, measure(agent, detailed=True, counter=counter)))
    if llm:
        llm.close()
    if args.json:
        print(json.dumps({name: ov.to_dict() for name, ov in rows}, indent=2))
    else:
        print(f"prompt overhead for config {cfg.name!r} (tokens the harness adds before the task)")
        print(format_overhead(rows))
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    store = _store(args)
    where, params = [], []
    if args.batch:
        where.append("r.batch_id = ?")
        params.append(store.resolve_batch(args.batch))
    if args.tag:
        where.append("f.tag = ?")
        params.append(args.tag)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    runs_clause = ("WHERE batch_id = ?", params[:1]) if args.batch else ("", [])
    n_runs = store.q(f"SELECT COUNT(*) AS n FROM runs {runs_clause[0]}", *runs_clause[1])[0]["n"]
    rows = store.q(
        f"""SELECT f.tag, COUNT(*) AS events, COUNT(DISTINCT f.run_id) AS runs
            FROM failures f JOIN runs r ON r.id = f.run_id {clause}
            GROUP BY f.tag ORDER BY runs DESC, events DESC""",
        *params,
    )
    print(f"{n_runs} runs")
    print(f"{'tag':<16}{'runs':>6}{'share':>8}{'events':>8}")
    for row in rows:
        share = f"{100 * row['runs'] / n_runs:.0f}%" if n_runs else "-"
        print(f"{row['tag']:<16}{row['runs']:>6}{share:>8}{row['events']:>8}")
    examples = store.q(
        f"""SELECT f.tag, f.run_id, f.turn_idx, f.detail FROM failures f
            JOIN runs r ON r.id = f.run_id {clause} ORDER BY f.id DESC LIMIT ?""",
        *params,
        args.limit,
    )
    if examples:
        print("\nrecent:")
        for e in examples:
            detail = (e["detail"] or "").replace("\n", " ")
            print(f"  {e['tag']:<15} {e['run_id']} t{e['turn_idx']}: {detail[:110]}")
    return 0


def _shadow(store: Store, run_id: str):
    """Shadow repository holding a run's checkpoints, reached through the run's executor."""
    from argus.config import config_from_dict
    from argus.executors import make_executor
    from argus.fsstate import ShadowGit

    row = store.run(run_id)
    cfg = config_from_dict(json.loads(row["config_json"]))
    cfg.executor.workdir = row["workspace"]
    ex = make_executor(cfg.executor)
    return ShadowGit(ex, cfg.checkpoint.shadow_root, cfg.checkpoint.excludes)


def _pick_checkpoint(rows: list, which: str) -> str:
    if which in ("baseline", "first"):
        return rows[0]["commit_sha"]
    if which in ("final", "last"):
        return rows[-1]["commit_sha"]
    try:
        return rows[int(which)]["commit_sha"]
    except (ValueError, IndexError):
        raise SystemExit(
            f"argus: no checkpoint {which!r} (use baseline, final or 0-{len(rows) - 1})"
        ) from None


def _run_checkpoints(args: argparse.Namespace) -> tuple[Store, str, list]:
    store = _store(args)
    run_id = store.resolve_run(args.run)
    rows = store.checkpoints(run_id)
    if not rows:
        raise SystemExit(f"argus: run {run_id} has no checkpoints")
    return store, run_id, rows


def cmd_checkpoints(args: argparse.Namespace) -> int:
    store, run_id, rows = _run_checkpoints(args)
    git = _shadow(store, run_id)
    prev = None
    print(f"checkpoints of run {run_id} (shadow repo {git.git_dir})")
    for i, r in enumerate(rows):
        changed = ""
        if prev and prev != r["tree_sha"]:
            changed = summarize(git.diff(prev, r["tree_sha"]), 5)
        elif prev:
            changed = "(no change)"
        turn = "-" if r["turn_idx"] is None else r["turn_idx"]
        print(f"{i:>3}  t{turn:<4} {r['commit_sha'][:10]}  {r['reason']:<18} {changed}")
        prev = r["tree_sha"]
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    store, run_id, rows = _run_checkpoints(args)
    git = _shadow(store, run_id)
    a = _pick_checkpoint(rows, args.frm)
    b = git.snapshot() if args.to == "now" else _pick_checkpoint(rows, args.to)
    if args.stat:
        for c in git.diff(a, b):
            print(c)
    else:
        sys.stdout.write(git.patch(a, b))
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    store, run_id, rows = _run_checkpoints(args)
    git = _shadow(store, run_id)
    target = _pick_checkpoint(rows, args.to)
    now = git.snapshot()
    plan = git.diff(now, git.tree_of(target))
    if not plan:
        print("workspace already matches that checkpoint")
        return 0
    verbs = {"D": "delete", "A": "restore", "M": "revert", "T": "revert"}
    for c in plan:
        print(f"{verbs.get(c.status, c.status)} {c.path}")
    if not args.yes:
        print(f"\n{len(plan)} changes; re-run with --yes to apply (current state is saved first)")
        return 1
    safety = git.commit(now, f"before restore of run {run_id} to {args.to}")
    git.update_ref(f"refs/argus/{run_id}-pre-restore", safety)
    git.restore(target)
    print(
        f"restored {len(plan)} paths; previous state saved as {safety[:10]} (refs/argus/{run_id}-pre-restore)"
    )
    return 0


def _progress(args: argparse.Namespace):
    return (lambda text: print(text, file=sys.stderr, flush=True)) if not args.quiet else None


def _reporter_factory(args: argparse.Namespace):
    from argus.console import ConsoleReporter

    if args.quiet or not args.verbose_runs:
        return None
    return lambda: ConsoleReporter(verbose=args.verbose)


def _finish_batch(store: Store, batch_id: str, outcomes: list, args: argparse.Namespace) -> int:
    from argus.report import format_report, summarize

    summary = summarize(store, batch_id)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_report(summary))
        kept = [o for o in outcomes if o.workdir and (args.keep or not o.passed)]
        if kept:
            print("\nworkspaces kept:")
            for o in kept:
                print(f"  {o.task} · {o.variant}: {o.workdir}")
    return 0 if all(o.passed for o in outcomes) else 1


def cmd_suite_run(args: argparse.Namespace) -> int:
    from argus.suite import SuiteRunner, load_suite

    suite = load_suite(args.suite)
    cfg = _config(args)
    store = Store(cfg.db_path())
    runner = SuiteRunner(
        suite,
        [(args.label or cfg.name, cfg)],
        store,
        repeat=args.repeat,
        task_ids=args.task,
        oracle=True if args.oracle else None,
        keep=args.keep,
        reporter_factory=_reporter_factory(args),
        progress=_progress(args),
    )
    batch_id, outcomes = runner.run()
    return _finish_batch(store, batch_id, outcomes, args)


def cmd_suite_list(args: argparse.Namespace) -> int:
    from argus.suite import load_suite

    suite = load_suite(args.suite)
    print(f"{suite.name}: {len(suite.tasks)} tasks" + (" (oracle)" if suite.oracle else ""))
    for t in suite.tasks:
        src = t.workspace or t.repo or "(empty workspace)"
        print(f"  {t.id:<24} check={t.check or '-'}  from={src}")
        print(f"  {'':<24} {t.prompt.strip().splitlines()[0][:90]}")
    return 0


def cmd_suite_add(args: argparse.Namespace) -> int:
    from argus.suite import append_task

    store = _store(args)
    run = store.run(store.resolve_run(args.run))
    workspace = args.workspace
    if workspace:
        workspace = str(Path(workspace).resolve())
    append_task(
        args.suite,
        {
            "id": args.id or f"task-{run['id']}",
            "prompt": run["task"],
            "workspace": workspace,
            "check": args.check,
        },
    )
    print(f"added task from run {run['id']} to {args.suite}")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    from argus.suite import SuiteRunner, load_suite

    suite = load_suite(args.suite)
    common = list(args.override or [])
    spec_a = model_overrides(args.model_a) if args.model_a else []
    spec_b = model_overrides(args.model_b) if args.model_b else []
    cfg_a = load_config(args.config_a, spec_a + common + list(args.override_a or []))
    cfg_b = load_config(args.config_b, spec_b + common + list(args.override_b or []))
    if args.db:
        cfg_a.log.db = cfg_b.log.db = args.db
    label_a, label_b = args.label_a or cfg_a.name, args.label_b or cfg_b.name
    if label_a == label_b:
        label_a, label_b = f"{label_a}-A", f"{label_b}-B"
    store = Store(cfg_a.db_path())
    runner = SuiteRunner(
        suite,
        [(label_a, cfg_a), (label_b, cfg_b)],
        store,
        repeat=args.repeat,
        task_ids=args.task,
        oracle=True if args.oracle else None,
        keep=args.keep,
        reporter_factory=_reporter_factory(args),
        progress=_progress(args),
        kind="ab",
    )
    batch_id, outcomes = runner.run()
    return _finish_batch(store, batch_id, outcomes, args)


def cmd_report(args: argparse.Namespace) -> int:
    from argus.report import format_report, summarize

    store = _store(args)
    batch_id = store.resolve_batch(args.batch)
    summary = summarize(store, batch_id)
    print(json.dumps(summary, indent=2) if args.json else format_report(summary))
    return 0


LOCAL_SERVERS = [
    ("llama_server", "http://127.0.0.1:8080/v1"),
    ("ollama", "http://127.0.0.1:11434/v1"),
    ("lmstudio", "http://127.0.0.1:1234/v1"),
    ("vllm", "http://127.0.0.1:8000/v1"),
]


def _k(n: int) -> str:
    if not n:
        return "-"
    return f"{n // 1000}k" if n >= 1000 else str(n)


def cmd_models(args: argparse.Namespace) -> int:
    from argus.profiles import load_profiles
    from argus.providers import KEY_ENV, make_provider

    if args.spec:
        return _model_detail(args)
    profiles = load_profiles()
    if args.json:
        print(json.dumps({n: p.to_dict() for n, p in profiles.items()}, indent=2))
        return 0
    head = f"{'profile':<22}{'provider':<11}{'ctx':>7}  {'protocol':<12}{'thinking':<22}{'$ in/out':>13}  source"
    print(head)
    for name, p in profiles.items():
        think = p.thinking + (f"/{p.effort}" if p.effort else "")
        price = (
            f"{p.pricing.get('input', 0):g}/{p.pricing.get('output', 0):g}" if p.pricing else "-"
        )
        print(
            f"{name:<22}{p.provider or 'local':<11}{_k(p.context_window):>7}  "
            f"{p.protocol or '-':<12}{think or '-':<22}{price:>13}  {p.source}"
        )
    print()
    keys = [n for names in KEY_ENV.values() for n in names if os.environ.get(n)]
    print("API keys set: " + (", ".join(keys) if keys else "none"))
    if args.detect:
        print("local servers:")
        for flavor, url in LOCAL_SERVERS:
            p = make_provider(flavor, base_url=url, connect_timeout=0.5, retries=0)
            d = p.detect()
            if d.models or d.context_window:
                print(
                    f"  {flavor:<13}{url:<28} ctx {_k(d.context_window):>6}  {', '.join(d.models)}"
                )
                for note in d.notes:
                    print(f"    note: {note}")
            else:
                print(f"  {flavor:<13}{url:<28} not running")
            p.close()
    return 0


def _model_detail(args: argparse.Namespace) -> int:
    from argus.providers import provider_from_config

    args.model = args.spec
    cfg = _config(args)
    m = cfg.model
    applied = getattr(cfg, "profile_applied", {}) or {}
    print(f"model:     {m.model}")
    print(f"provider:  {m.provider}  {m.base_url or '(default URL)'}")
    print(f"profile:   {m.profile or '(none matched)'}")
    for key, value in applied.items():
        print(f"  {key} = {json.dumps(value)}")
    print(f"protocol:  {cfg.agent.protocol}")
    if args.detect:
        p = provider_from_config(m, require_key=False)
        d = p.detect()
        print(f"detected:  {p.describe()}")
        print(f"  context window {d.context_window or '?'}  max output {d.max_output or '?'}")
        if d.capabilities:
            print(f"  capabilities: {', '.join(sorted(d.capabilities))}")
        if d.pricing:
            print(f"  pricing: {json.dumps(d.pricing)}")
        print(f"  protocols: {', '.join(p.protocols())}")
        for note in d.notes:
            print(f"  note: {note}")
        p.close()
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from argus.providers import provider_from_config
    from argus.report import format_report, summarize
    from argus.suite import load_suite
    from argus.tune import format_ranking, save, tune

    args.model = args.spec
    cfg = _config(args)
    probe = provider_from_config(cfg.model)  # what can this provider serve?
    supported = list(probe.protocols())
    probe.close()
    protocols = [p for p in (args.protocols or ",".join(supported)).split(",") if p]
    unknown = [p for p in protocols if p not in supported]
    if unknown:
        print(f"argus: {', '.join(unknown)} not supported here; choose from {supported}")
        return 2
    suite = load_suite(args.suite) if args.suite else None
    store = Store(cfg.db_path())
    print(f"tuning {cfg.model.model} on {', '.join(protocols)}", file=sys.stderr)
    batch_id, ranked, suite = tune(
        cfg,
        protocols,
        store,
        suite=suite,
        repeat=args.repeat,
        thinking=args.thinking,
        task_ids=args.task,
        reporter_factory=_reporter_factory(args),
        progress=_progress(args),
    )
    print(format_report(summarize(store, batch_id)))
    print()
    print(format_ranking(ranked))
    winner = ranked[0]
    if winner.passes == 0:
        print("\nno variant passed a task; nothing saved")
        return 1
    if args.dry_run:
        print(f"\nbest: {winner.label} (dry run, nothing saved)")
        return 0
    path, name = save(cfg, ranked, batch_id, suite)
    print(f"\nsaved protocol={winner.protocol} to {path} [profiles.{name}]")
    return 0


def cmd_mock_server(args: argparse.Namespace) -> int:
    from argus.mock import MockServer, Script

    script = Script.load(args.script) if args.script else Script([], on_exhausted="final")
    server = MockServer(
        script, host=args.host, port=args.port, n_ctx=args.n_ctx, chunk_delay=args.chunk_delay
    )
    print(f"mock llama-server listening on {server.url}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--repeat", type=int, default=1, help="runs per task and variant")
    p.add_argument("-t", "--task", action="append", help="only this task id (repeatable)")
    p.add_argument("--oracle", action="store_true", help="run the check after each mutating turn")
    p.add_argument(
        "--keep", action="store_true", help="keep all workspaces (default: keep failures)"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="with --runs: stream model output")
    p.add_argument(
        "--runs", dest="verbose_runs", action="store_true", help="show each run's progress"
    )
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def _add_tui_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-w", "--workdir", help="workspace directory")
    _add_config_args(p)
    p.add_argument(
        "--continue", dest="cont", action="store_true", help="continue the latest session here"
    )
    p.add_argument("--session", help="continue this session")
    p.add_argument("--plan", action="store_true", help="start in plan mode")
    p.add_argument("--approval", choices=["read-only", "ask", "auto", "full"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="argus", description="Model-agnostic, headless coding agent and measurement harness."
    )
    from argus import __version__

    p.add_argument("--version", action="version", version=f"argus {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="run one task")
    r.add_argument("task", nargs="?", help="task text, or - for stdin")
    r.add_argument("-f", "--task-file", help="read the task from a file")
    r.add_argument("-w", "--workdir", help="workspace directory (local executor)")
    _add_config_args(r)
    r.add_argument("-v", "--verbose", action="store_true", help="stream reasoning and content")
    r.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    r.add_argument("--json", action="store_true", help="print a JSON result on stdout")
    r.add_argument("--record", metavar="DIR", help="save every model API exchange as a fixture")
    r.add_argument(
        "--approval",
        choices=["read-only", "ask", "auto", "full"],
        help="approval policy (default: agent.approval, normally auto)",
    )
    r.add_argument("-y", "--yes", action="store_true", help="approve everything argus would ask")
    r.add_argument(
        "--plan", action="store_true", help="plan mode: explore read-only, answer with a plan"
    )
    r.add_argument(
        "--continue",
        dest="cont",
        action="store_true",
        help="continue the latest session in this workspace (any model)",
    )
    r.add_argument("--session", help="continue this session (id prefix)")
    r.set_defaults(fn=cmd_run)

    tu_ = sub.add_parser("tui", help="the terminal UI (also: argus with no arguments)")
    _add_tui_args(tu_)
    tu_.set_defaults(fn=cmd_tui)

    it = sub.add_parser("init", help="write ~/.config/argus/config.toml and directories")
    it.add_argument("--force", action="store_true", help="overwrite an existing config")
    it.set_defaults(fn=cmd_init)

    dr = sub.add_parser("doctor", help="what works on this machine")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(fn=cmd_doctor)

    ss = sub.add_parser("sessions", help="list recent sessions")
    ss.add_argument("-n", "--limit", type=int, default=20)
    ss.add_argument("-c", "--config")
    ss.add_argument("--db")
    ss.set_defaults(fn=cmd_sessions)

    rs_ = sub.add_parser("resume", help="continue a session: argus resume [SESSION] TASK")
    rs_.add_argument("session_id", nargs="?", help="session id prefix (default: latest here)")
    rs_.add_argument("task", nargs="?", help="the next message (or - for stdin)")
    rs_.add_argument("-f", "--task-file")
    rs_.add_argument("-w", "--workdir")
    _add_config_args(rs_)
    rs_.add_argument("-v", "--verbose", action="store_true")
    rs_.add_argument("-q", "--quiet", action="store_true")
    rs_.add_argument("--json", action="store_true")
    rs_.add_argument("--record", metavar="DIR")
    rs_.add_argument("--approval", choices=["read-only", "ask", "auto", "full"])
    rs_.add_argument("-y", "--yes", action="store_true")
    rs_.add_argument("--plan", action="store_true")
    rs_.set_defaults(fn=cmd_resume)

    ls = sub.add_parser("runs", help="list recent runs")
    ls.add_argument("-n", "--limit", type=int, default=20)
    ls.add_argument("-c", "--config")
    ls.add_argument("--db")
    ls.set_defaults(fn=cmd_runs)

    sh = sub.add_parser("show", help="show a run transcript (id prefix or 'last')")
    sh.add_argument("run", nargs="?", default="last")
    sh.add_argument("--full", action="store_true", help="do not clip long text")
    sh.add_argument("-c", "--config")
    sh.add_argument("--db")
    sh.set_defaults(fn=cmd_show)

    fl = sub.add_parser("failures", help="failure tag summary across runs")
    fl.add_argument("--batch", help="restrict to a suite/A-B batch")
    fl.add_argument("--tag", help="only this tag")
    fl.add_argument("-n", "--limit", type=int, default=15, help="recent examples to show")
    fl.add_argument("-c", "--config")
    fl.add_argument("--db")
    fl.set_defaults(fn=cmd_failures)

    su = sub.add_parser("suite", help="task suites: run, list, add")
    su_sub = su.add_subparsers(dest="suite_command", required=True)
    sr = su_sub.add_parser("run", help="run every task of a suite in a fresh workspace")
    sr.add_argument("suite", help="suite TOML file")
    _add_config_args(sr)
    _add_batch_args(sr)
    sr.add_argument("--label", help="variant label (default: config name)")
    sr.set_defaults(fn=cmd_suite_run)
    sl = su_sub.add_parser("list", help="show a suite's tasks")
    sl.add_argument("suite")
    sl.set_defaults(fn=cmd_suite_list)
    sa = su_sub.add_parser("add", help="append a logged run's task to a suite")
    sa.add_argument("suite")
    sa.add_argument("--run", default="last", help="run id (default: last)")
    sa.add_argument("--id", help="task id")
    sa.add_argument("--check", help="command that passes (exit 0) when the task is done")
    sa.add_argument("--workspace", help="directory to copy as the task's starting workspace")
    sa.add_argument("--db")
    sa.add_argument("-c", "--config")
    sa.set_defaults(fn=cmd_suite_add)

    ab = sub.add_parser("ab", help="run a suite under two configs and compare")
    ab.add_argument("config_a", help="config A (TOML)")
    ab.add_argument("config_b", help="config B (TOML)")
    ab.add_argument("--suite", required=True)
    ab.add_argument(
        "-o", "--override", action="append", metavar="KEY=VALUE", help="override for both"
    )
    ab.add_argument("--oa", "--override-a", dest="override_a", action="append", metavar="KEY=VALUE")
    ab.add_argument("--ob", "--override-b", dest="override_b", action="append", metavar="KEY=VALUE")
    ab.add_argument("--ma", "--model-a", dest="model_a", metavar="SPEC", help="model for A")
    ab.add_argument("--mb", "--model-b", dest="model_b", metavar="SPEC", help="model for B")
    ab.add_argument("--label-a")
    ab.add_argument("--label-b")
    ab.add_argument("--db")
    _add_batch_args(ab)
    ab.set_defaults(fn=cmd_ab)

    rp = sub.add_parser("report", help="report for a suite or A/B batch")
    rp.add_argument("batch", nargs="?", default="last")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--db")
    rp.add_argument("-c", "--config")
    rp.set_defaults(fn=cmd_report)

    ck = sub.add_parser("checkpoints", help="list a run's workspace checkpoints")
    ck.add_argument("run", nargs="?", default="last")
    ck.add_argument("--db")
    ck.add_argument("-c", "--config")
    ck.set_defaults(fn=cmd_checkpoints)

    df = sub.add_parser("diff", help="diff the workspace between two checkpoints of a run")
    df.add_argument("run", nargs="?", default="last")
    df.add_argument(
        "--from", dest="frm", default="baseline", help="baseline | final | N (default baseline)"
    )
    df.add_argument("--to", default="final", help="baseline | final | N | now (default final)")
    df.add_argument("--stat", action="store_true", help="only list changed files")
    df.add_argument("--db")
    df.add_argument("-c", "--config")
    df.set_defaults(fn=cmd_diff)

    rs = sub.add_parser("restore", help="restore the workspace to a checkpoint of a run")
    rs.add_argument("run", nargs="?", default="last")
    rs.add_argument("--to", default="baseline", help="baseline | final | N (default baseline)")
    rs.add_argument("--yes", action="store_true", help="apply (otherwise only show the plan)")
    rs.add_argument("--db")
    rs.add_argument("-c", "--config")
    rs.set_defaults(fn=cmd_restore)

    ov = sub.add_parser("overhead", help="measure system prompt + tool schema token overhead")
    ov.add_argument("-w", "--workdir", help="workspace (affects AGENTS.md, skills and the prompt)")
    _add_config_args(ov)
    ov.add_argument("--all", action="store_true", help="compare all tool-call protocols")
    ov.add_argument("--offline", action="store_true", help="estimate without contacting the server")
    ov.add_argument("--json", action="store_true")
    ov.set_defaults(fn=cmd_overhead)

    mo = sub.add_parser("models", help="model profiles; SPEC shows what applies to one model")
    mo.add_argument("spec", nargs="?", help="provider/model or profile name")
    mo.add_argument("--detect", action="store_true", help="ask servers/APIs what they report")
    mo.add_argument("--json", action="store_true")
    mo.add_argument("-c", "--config")
    mo.add_argument("-o", "--override", action="append", metavar="KEY=VALUE")
    mo.set_defaults(fn=cmd_models)

    tu = sub.add_parser("tune", help="find the best protocol for a model and save it")
    tu.add_argument("spec", help="provider/model or profile name")
    tu.add_argument("--suite", help="suite TOML (default: argus's built-in tuning suite)")
    tu.add_argument("--protocols", help="comma-separated (default: all the provider supports)")
    tu.add_argument("--thinking", action="store_true", help="also try thinking on and off")
    tu.add_argument("--dry-run", action="store_true", help="report without saving")
    tu.add_argument("-c", "--config")
    tu.add_argument("-o", "--override", action="append", metavar="KEY=VALUE")
    tu.add_argument("--db")
    tu.add_argument("-n", "--repeat", type=int, default=2)
    tu.add_argument("-t", "--task", action="append", help="only these task ids")
    tu.add_argument("-v", "--verbose", action="store_true", help="with --runs: stream output")
    tu.add_argument("--runs", dest="verbose_runs", action="store_true", help="show each run")
    tu.add_argument("-q", "--quiet", action="store_true")
    tu.set_defaults(fn=cmd_tune)

    ms = sub.add_parser("mock-server", help="serve a scripted mock of llama-server")
    ms.add_argument("--script", help="JSON script file")
    ms.add_argument("--host", default="127.0.0.1")
    ms.add_argument("--port", type=int, default=8080)
    ms.add_argument("--n-ctx", type=int, default=32768)
    ms.add_argument("--chunk-delay", type=float, default=0.0)
    ms.set_defaults(fn=cmd_mock_server)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # plain `argus`: the TUI when it is installed
        from argus.tui import available

        if not available() or not sys.stdout.isatty():
            parser.print_help()
            return 0
        args = parser.parse_args(["tui"])
    try:
        return int(args.fn(args) or 0)
    except ConfigError as e:
        print(f"argus: config error: {e}", file=sys.stderr)
        return 2
    except (SuiteError, KeyError) as e:
        print(f"argus: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
