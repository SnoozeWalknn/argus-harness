"""argus command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus.config import Config, ConfigError, load_config
from argus.fsstate import summarize
from argus.store import Store


def _config(args: argparse.Namespace) -> Config:
    overrides = list(getattr(args, "override", None) or [])
    if getattr(args, "workdir", None):
        overrides.append(f"executor.workdir={json.dumps(str(Path(args.workdir).resolve()))}")
    if getattr(args, "db", None):
        overrides.append(f"log.db={json.dumps(args.db)}")
    return load_config(getattr(args, "config", None), overrides)


def _store(args: argparse.Namespace) -> Store:
    if getattr(args, "db", None):
        return Store(args.db)
    cfg = load_config(getattr(args, "config", None)) if getattr(args, "config", None) else Config()
    return Store(cfg.db_path())


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="TOML config file")
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
    cfg = _config(args)
    reporter = ConsoleReporter(verbose=args.verbose) if not args.quiet else None
    agent = Agent(cfg, reporter=reporter)
    try:
        result = agent.run(task)
    finally:
        agent.close()
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
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
    return 0 if result.status == "completed" else 1


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
    from argus.llm import LLMClient
    from argus.tokens import TokenCounter, format_overhead, measure

    cfg = _config(args)
    protocols = PROTOCOLS if args.all else [cfg.agent.protocol]
    m = cfg.model
    llm = (
        None if args.offline else LLMClient(m.base_url, m.api_key, m.timeout, m.connect_timeout, 0)
    )
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="argus", description="Headless coding-agent harness for llama-server."
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one task")
    r.add_argument("task", nargs="?", help="task text, or - for stdin")
    r.add_argument("-f", "--task-file", help="read the task from a file")
    r.add_argument("-w", "--workdir", help="workspace directory (local executor)")
    _add_config_args(r)
    r.add_argument("-v", "--verbose", action="store_true", help="stream reasoning and content")
    r.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    r.add_argument("--json", action="store_true", help="print a JSON result on stdout")
    r.set_defaults(fn=cmd_run)

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

    ms = sub.add_parser("mock-server", help="serve a scripted mock of llama-server")
    ms.add_argument("--script", help="JSON script file")
    ms.add_argument("--host", default="127.0.0.1")
    ms.add_argument("--port", type=int, default=8080)
    ms.add_argument("--n-ctx", type=int, default=32768)
    ms.add_argument("--chunk-delay", type=float, default=0.0)
    ms.set_defaults(fn=cmd_mock_server)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except ConfigError as e:
        print(f"argus: config error: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
