"""Batch reports: pass rates, cost, failure tags, and A/B comparisons."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from argus.store import Store


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate."""
    if n == 0:
        return 0.0, 0.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def run_passed(row: Any) -> bool:
    if row["check_passed"] is not None:
        return bool(row["check_passed"])
    return row["status"] == "completed"


@dataclass
class VariantStats:
    label: str
    config: str = ""
    runs: int = 0
    passes: int = 0
    turns: list[float] = field(default_factory=list)
    gen_tokens: list[float] = field(default_factory=list)
    prompt_tokens: list[float] = field(default_factory=list)
    max_ctx: list[float] = field(default_factory=list)
    wall_s: list[float] = field(default_factory=list)
    overhead: list[float] = field(default_factory=list)
    statuses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tags: dict[str, int] = field(default_factory=lambda: defaultdict(int))  # runs with the tag

    def mean(self, xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def to_dict(self) -> dict[str, Any]:
        lo, hi = wilson(self.passes, self.runs)
        return {
            "label": self.label,
            "config": self.config,
            "runs": self.runs,
            "passes": self.passes,
            "pass_rate": self.passes / self.runs if self.runs else 0.0,
            "ci95": [lo, hi],
            "mean_turns": self.mean(self.turns),
            "mean_gen_tokens": self.mean(self.gen_tokens),
            "mean_prompt_tokens": self.mean(self.prompt_tokens),
            "mean_max_context": self.mean(self.max_ctx),
            "mean_wall_s": self.mean(self.wall_s),
            "overhead_tokens": self.mean(self.overhead),
            "statuses": dict(self.statuses),
            "failure_runs": dict(self.tags),
        }


def summarize(store: Store, batch_id: str) -> dict[str, Any]:
    batch = store.q("SELECT * FROM batches WHERE id = ?", batch_id)[0]
    meta = json.loads(batch["meta_json"] or "{}")
    rows = store.runs(batch_id=batch_id)
    variants: dict[str, VariantStats] = {}
    per_task: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for r in rows:
        label = r["variant"] or r["config_name"] or "?"
        v = variants.setdefault(
            label, VariantStats(label, f"{r['config_name']} ({r['config_hash']})")
        )
        ok = run_passed(r)
        v.runs += 1
        v.passes += ok
        v.turns.append(r["n_turns"] or 0)
        v.gen_tokens.append(r["completion_tokens"] or 0)
        v.prompt_tokens.append(r["prompt_tokens"] or 0)
        v.max_ctx.append(r["max_context_tokens"] or 0)
        v.wall_s.append((r["wall_ms"] or 0) / 1000)
        if r["overhead_tokens"] is not None:
            v.overhead.append(r["overhead_tokens"])
        v.statuses[r["status"]] += 1
        for tag in {f["tag"] for f in store.failures(r["id"])}:
            v.tags[tag] += 1
        per_task[r["task_id"]][label].append(ok)
        paired[(r["task_id"], r["rep"] or 0)][label] = ok
    declared = list((meta.get("variants") or {}).keys())  # the order the user gave, not run order
    variants = dict(
        sorted(variants.items(), key=lambda kv: declared.index(kv[0]) if kv[0] in declared else 99)
    )
    out: dict[str, Any] = {
        "batch": batch_id,
        "kind": batch["kind"],
        "name": batch["name"],
        "meta": meta,
        "variants": [v.to_dict() for v in variants.values()],
        "tasks": {t: {lab: [sum(x), len(x)] for lab, x in d.items()} for t, d in per_task.items()},
    }
    labels = list(variants)
    if len(labels) == 2:
        a, b = labels
        only_a = sum(1 for p in paired.values() if p.get(a) is True and p.get(b) is False)
        only_b = sum(1 for p in paired.values() if p.get(a) is False and p.get(b) is True)
        out["paired"] = {
            "a": a,
            "b": b,
            "pairs": sum(1 for p in paired.values() if a in p and b in p),
            "only_a_passed": only_a,
            "only_b_passed": only_b,
            "mcnemar_p": mcnemar_exact(only_a, only_b),
        }
    return out


def format_report(s: dict[str, Any]) -> str:
    lines = [f"batch {s['batch']} ({s['kind']}: {s['name']})"]
    meta = s.get("meta") or {}
    if meta:
        lines.append(
            f"  {len(meta.get('tasks', []))} tasks × {len(s['variants'])} variants × {meta.get('repeat', 1)} repeats"
            + ("  (completion oracle on)" if meta.get("oracle") else "")
        )
    head = f"{'variant':<14}{'pass':>8}{'rate':>7}{'95% CI':>14}{'turns':>7}{'gen tok':>9}{'prompt Σ':>10}{'max ctx':>9}{'wall s':>8}{'overhead':>9}"
    lines += ["", head]
    for v in s["variants"]:
        lo, hi = v["ci95"]
        lines.append(
            f"{v['label']:<14}{v['passes']:>4}/{v['runs']:<3}{100 * v['pass_rate']:>6.0f}%"
            f"{f'[{100 * lo:.0f}-{100 * hi:.0f}%]':>14}{v['mean_turns']:>7.1f}{v['mean_gen_tokens']:>9.0f}"
            f"{v['mean_prompt_tokens']:>10.0f}{v['mean_max_context']:>9.0f}{v['mean_wall_s']:>8.1f}"
            f"{v['overhead_tokens']:>9.0f}"
        )
    lines.append("")
    lines.append("failure tags (runs affected):")
    for v in s["variants"]:
        tags = ", ".join(f"{t}:{n}" for t, n in sorted(v["failure_runs"].items())) or "none"
        statuses = ", ".join(f"{k}:{n}" for k, n in sorted(v["statuses"].items()))
        lines.append(f"  {v['label']:<12} {tags}   [{statuses}]")
    labels = [v["label"] for v in s["variants"]]
    lines += ["", "per task:", "  " + f"{'task':<24}" + "".join(f"{lab:>12}" for lab in labels)]
    for task, res in s["tasks"].items():
        cells = "".join(
            f"{f'{res[lab][0]}/{res[lab][1]}' if lab in res else '-':>12}" for lab in labels
        )
        lines.append(f"  {task:<24}{cells}")
    if "paired" in s:
        p = s["paired"]
        lines += [
            "",
            f"paired runs: {p['pairs']}; only {p['a']} passed: {p['only_a_passed']}, "
            f"only {p['b']} passed: {p['only_b_passed']}; exact McNemar p = {p['mcnemar_p']:.3f}",
        ]
        if p["pairs"] < 10:
            lines.append(
                "  (few pairs: treat differences as anecdotal; use --repeat and more tasks)"
            )
    return "\n".join(lines)
