"""argus command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus.config import Config, ConfigError, load_config
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
