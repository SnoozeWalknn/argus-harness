"""SQLite run log.

Committed after every write so a crash leaves a readable partial log.
``messages`` is append-only and ``turns.context_ids`` lists the message ids sent
in each request, so the exact prompt of every turn can be rebuilt without
storing a copy of the whole history per turn.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- suite | ab
    name TEXT,
    meta_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    batch_id TEXT REFERENCES batches(id),
    variant TEXT,                    -- A/B label or config name
    task_id TEXT,
    rep INTEGER,                     -- repetition index within a batch
    task TEXT NOT NULL,
    config_name TEXT,
    config_hash TEXT,
    config_json TEXT,
    workspace TEXT,
    executor TEXT,
    protocol TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- running|completed|failed|error|interrupted
    final_answer TEXT,
    n_turns INTEGER DEFAULT 0,
    n_tool_calls INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,       -- summed over turns (tokens the server evaluated or reused)
    completion_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    max_context_tokens INTEGER DEFAULT 0,  -- largest single prompt
    overhead_tokens INTEGER,               -- system prompt + tool schemas
    overhead_json TEXT,
    context_window INTEGER,
    wall_ms REAL,
    llm_ms REAL,
    tool_ms REAL,
    check_passed INTEGER,
    check_output TEXT,
    error TEXT,
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER,
    role TEXT NOT NULL,
    content TEXT,
    reasoning TEXT,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    kind TEXT DEFAULT 'normal',      -- normal|feedback|summary|masked
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    attempt INTEGER DEFAULT 0,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    content_tokens INTEGER,
    reasoning TEXT,
    content TEXT,
    tool_calls_json TEXT,
    finish_reason TEXT,
    aborted TEXT,
    ttft_ms REAL,
    prompt_ms REAL,
    gen_ms REAL,
    total_ms REAL,
    prompt_tps REAL,
    gen_tps REAL,
    max_tokens INTEGER,
    context_ids TEXT,
    raw_response TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    call_id TEXT,
    name TEXT,
    args_json TEXT,
    raw_args TEXT,
    ok INTEGER,
    error TEXT,
    result TEXT,
    result_chars INTEGER,
    truncated INTEGER,
    duration_ms REAL,
    meta_json TEXT,
    fs_changes_json TEXT,
    checkpoint TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER,
    tag TEXT NOT NULL,
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER,
    tool_call_idx INTEGER,
    commit_sha TEXT NOT NULL,
    tree_sha TEXT NOT NULL,
    reason TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS skill_invocations (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER,
    skill TEXT NOT NULL,
    path TEXT,
    via TEXT,                        -- tool | read
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS compactions (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    turn_idx INTEGER,
    stage TEXT,                      -- mask | summarize
    tokens_before INTEGER,
    tokens_after INTEGER,
    messages_affected INTEGER,
    duration_ms REAL,
    model TEXT,
    summary TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    turn_idx INTEGER,
    kind TEXT NOT NULL,
    data_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id, idx);
CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, turn_idx);
CREATE INDEX IF NOT EXISTS idx_failures_run ON failures(run_id);
CREATE INDEX IF NOT EXISTS idx_failures_tag ON failures(tag);
"""


# Applied in order to databases older than the version; SCHEMA above is version 1.
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE runs ADD COLUMN provider TEXT",
        "ALTER TABLE runs ADD COLUMN cache_read_tokens INTEGER DEFAULT 0",
        "ALTER TABLE runs ADD COLUMN cache_write_tokens INTEGER DEFAULT 0",
        "ALTER TABLE runs ADD COLUMN cost_usd REAL",
        "ALTER TABLE turns ADD COLUMN provider TEXT",
        "ALTER TABLE turns ADD COLUMN model TEXT",
        "ALTER TABLE turns ADD COLUMN cache_write_tokens INTEGER",
        "ALTER TABLE turns ADD COLUMN cost_usd REAL",
        "ALTER TABLE messages ADD COLUMN replay_json TEXT",
    ],
}


def new_id() -> str:
    return time.strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _j(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute("PRAGMA foreign_keys=ON")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} has schema v{version}; this argus knows v{SCHEMA_VERSION}"
                )
            self.db.executescript(SCHEMA)
            self._migrate(max(version, 1))
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.db.commit()

    def _migrate(self, version: int) -> None:
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for stmt in MIGRATIONS.get(target, []):
                try:
                    self.db.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e):
                        raise

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        row = {k: v for k, v in row.items() if v is not None}
        row.setdefault("created_at", time.time())
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self.lock:
            cur = self.db.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values())
            )
            self.db.commit()
            return int(cur.lastrowid or 0)

    # -- writes ----------------------------------------------------------------------------

    def new_batch(self, kind: str, name: str = "", meta: dict[str, Any] | None = None) -> str:
        bid = "b" + new_id()
        with self.lock:
            self.db.execute(
                "INSERT INTO batches (id, kind, name, meta_json, created_at) VALUES (?,?,?,?,?)",
                (bid, kind, name, _j(meta or {}), time.time()),
            )
            self.db.commit()
        return bid

    def start_run(self, run_id: str, **fields: Any) -> None:
        fields = {**fields, "id": run_id, "started_at": time.time()}
        with self.lock:
            cols = ", ".join(fields)
            marks = ", ".join("?" for _ in fields)
            self.db.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(fields.values()))
            self.db.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE runs SET {sets} WHERE id = ?", [*fields.values(), run_id])
            self.db.commit()

    def add_message(
        self, run_id: str, turn_idx: int | None, msg: dict[str, Any], kind: str = "normal"
    ) -> int:
        return self._insert(
            "messages",
            {
                "run_id": run_id,
                "turn_idx": turn_idx,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "reasoning": msg.get("reasoning_content"),
                "tool_calls_json": _j(msg.get("tool_calls")),
                "tool_call_id": msg.get("tool_call_id"),
                "replay_json": _j(msg.get("replay")),
                "kind": kind,
            },
        )

    def add_turn(self, run_id: str, idx: int, **fields: Any) -> int:
        for key in ("context_ids", "raw_response", "tool_calls_json"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = _j(fields[key])
        return self._insert("turns", {"run_id": run_id, "idx": idx, **fields})

    def add_tool_call(self, run_id: str, turn_idx: int, idx: int, **fields: Any) -> int:
        for key in ("args_json", "meta_json", "fs_changes_json"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = _j(fields[key])
        return self._insert(
            "tool_calls", {"run_id": run_id, "turn_idx": turn_idx, "idx": idx, **fields}
        )

    def add_failure(self, run_id: str, turn_idx: int | None, tag: str, detail: str = "") -> int:
        return self._insert(
            "failures", {"run_id": run_id, "turn_idx": turn_idx, "tag": tag, "detail": detail}
        )

    def add_event(
        self, run_id: str | None, turn_idx: int | None, kind: str, data: Any = None
    ) -> int:
        return self._insert(
            "events", {"run_id": run_id, "turn_idx": turn_idx, "kind": kind, "data_json": _j(data)}
        )

    def add_checkpoint(
        self,
        run_id: str,
        turn_idx: int,
        tool_call_idx: int | None,
        commit_sha: str,
        tree_sha: str,
        reason: str = "",
    ) -> int:
        return self._insert(
            "checkpoints",
            {
                "run_id": run_id,
                "turn_idx": turn_idx,
                "tool_call_idx": tool_call_idx,
                "commit_sha": commit_sha,
                "tree_sha": tree_sha,
                "reason": reason,
            },
        )

    def add_skill_invocation(
        self, run_id: str, turn_idx: int, skill: str, path: str, via: str
    ) -> int:
        return self._insert(
            "skill_invocations",
            {"run_id": run_id, "turn_idx": turn_idx, "skill": skill, "path": path, "via": via},
        )

    def add_compaction(self, run_id: str, turn_idx: int, **fields: Any) -> int:
        return self._insert("compactions", {"run_id": run_id, "turn_idx": turn_idx, **fields})

    # -- reads -----------------------------------------------------------------------------

    def q(self, sql: str, *params: Any) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.db.execute(sql, params))

    def resolve_run(self, prefix: str) -> str:
        if prefix in ("last", "latest", "-"):
            rows = self.q("SELECT id FROM runs ORDER BY started_at DESC LIMIT 1")
        else:
            rows = self.q(
                "SELECT id FROM runs WHERE id LIKE ? ORDER BY started_at DESC", prefix + "%"
            )
            if not rows:
                rows = self.q(
                    "SELECT id FROM runs WHERE id LIKE ? ORDER BY started_at DESC",
                    "%" + prefix + "%",
                )
        if not rows:
            raise KeyError(f"no run matching {prefix!r}")
        if len(rows) > 1 and rows[0]["id"] != prefix:
            ids = ", ".join(r["id"] for r in rows[:5])
            raise KeyError(f"{prefix!r} is ambiguous: {ids}")
        return rows[0]["id"]

    def resolve_batch(self, prefix: str) -> str:
        if prefix in ("last", "latest", "-"):
            rows = self.q("SELECT id FROM batches ORDER BY created_at DESC LIMIT 1")
        else:
            rows = self.q(
                "SELECT id FROM batches WHERE id LIKE ? ORDER BY created_at DESC",
                "%" + prefix + "%",
            )
        if not rows:
            raise KeyError(f"no batch matching {prefix!r}")
        return rows[0]["id"]

    def run(self, run_id: str) -> sqlite3.Row:
        rows = self.q("SELECT * FROM runs WHERE id = ?", run_id)
        if not rows:
            raise KeyError(run_id)
        return rows[0]

    def runs(self, limit: int = 20, batch_id: str | None = None) -> list[sqlite3.Row]:
        if batch_id:
            return self.q("SELECT * FROM runs WHERE batch_id = ? ORDER BY started_at", batch_id)
        return self.q("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", limit)

    def turns(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM turns WHERE run_id = ? ORDER BY idx, attempt, id", run_id)

    def messages(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM messages WHERE run_id = ? ORDER BY id", run_id)

    def tool_calls(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY turn_idx, idx", run_id)

    def failures(self, run_id: str | None = None) -> list[sqlite3.Row]:
        if run_id:
            return self.q("SELECT * FROM failures WHERE run_id = ? ORDER BY id", run_id)
        return self.q("SELECT * FROM failures ORDER BY id")

    def events(self, run_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return self.q(
                "SELECT * FROM events WHERE run_id = ? AND kind = ? ORDER BY id", run_id, kind
            )
        return self.q("SELECT * FROM events WHERE run_id = ? ORDER BY id", run_id)

    def checkpoints(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM checkpoints WHERE run_id = ? ORDER BY id", run_id)

    def skill_invocations(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM skill_invocations WHERE run_id = ? ORDER BY id", run_id)

    def compactions(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM compactions WHERE run_id = ? ORDER BY id", run_id)
