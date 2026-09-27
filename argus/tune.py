"""``argus tune MODEL``: find the settings that work best for a model and save them.

Runs a suite once per tool-call protocol the provider can serve (optionally
with thinking on and off), as one batch, ranks the variants, and writes the
winner into the model's user profile together with a ``tuned`` record, so every
later run of that model starts from what was measured rather than guessed.

Ranking: Wilson lower bound of the pass rate, then pass rate, then failure
tags per run, then generated tokens, then wall time.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from argus.config import Config, apply_override
from argus.profiles import save_user_profile
from argus.report import summarize
from argus.store import Store
from argus.suite import Suite, SuiteRunner, load_suite


def builtin_suite() -> Suite:
    return load_suite(Path(str(resources.files("argus.data").joinpath("tune/suite.toml"))))


@dataclass
class Ranked:
    label: str
    protocol: str
    enable_thinking: bool | None
    passes: int
    runs: int
    lower: float
    failures_per_run: float
    gen_tokens: float
    wall_s: float
    cost: float | None

    @property
    def rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


def variants(
    base: Config, protocols: list[str], thinking: bool = False
) -> list[tuple[str, Config, str, bool | None]]:
    """(label, config, protocol, enable_thinking) for every combination to try."""
    out = []
    for proto in protocols:
        for think in (True, False) if thinking else (None,):
            cfg = copy.deepcopy(base)
            apply_override(cfg, f'agent.protocol="{proto}"')
            label = proto
            if think is not None:
                apply_override(cfg, f"model.enable_thinking={'true' if think else 'false'}")
                label += "+think" if think else "-think"
            cfg.name = label
            out.append((label, cfg, proto, think))
    return out


def rank(summary: dict[str, Any], meta: dict[str, tuple[str, bool | None]]) -> list[Ranked]:
    ranked = []
    for v in summary["variants"]:
        proto, think = meta.get(v["label"], (v["label"], None))
        runs = v["runs"] or 1
        ranked.append(
            Ranked(
                label=v["label"],
                protocol=proto,
                enable_thinking=think,
                passes=v["passes"],
                runs=v["runs"],
                lower=v["ci95"][0],
                failures_per_run=sum(v["failure_runs"].values()) / runs,
                gen_tokens=v["mean_gen_tokens"],
                wall_s=v["mean_wall_s"],
                cost=v.get("mean_cost_usd"),
            )
        )
    ranked.sort(key=lambda r: (-r.lower, -r.rate, r.failures_per_run, r.gen_tokens, r.wall_s))
    return ranked


def format_ranking(ranked: list[Ranked]) -> str:
    lines = [
        f"{'':2}{'variant':<18}{'pass':>8}{'rate':>7}{'lower95':>9}{'fail/run':>10}{'gen tok':>9}{'wall s':>8}"
    ]
    for i, r in enumerate(ranked):
        mark = "→ " if i == 0 else "  "
        lines.append(
            f"{mark}{r.label:<18}{r.passes:>4}/{r.runs:<3}{100 * r.rate:>6.0f}%{100 * r.lower:>8.0f}%"
            f"{r.failures_per_run:>10.2f}{r.gen_tokens:>9.0f}{r.wall_s:>8.1f}"
        )
    return "\n".join(lines)


def profile_updates(
    cfg: Config, winner: Ranked, ranked: list[Ranked], batch_id: str, suite: Suite
) -> tuple[str, dict[str, Any]]:
    """The user profile to write: one per model id, extending the matched family profile."""
    model = cfg.model.model
    name = model
    updates: dict[str, Any] = {"match": [model], "protocol": winner.protocol}
    family = cfg.model.profile
    if family and family != name:
        updates["extends"] = family
    if cfg.model.provider not in ("", "auto"):
        updates["provider"] = cfg.model.provider
    if winner.enable_thinking is not None:
        updates["enable_thinking"] = winner.enable_thinking
    updates["tuned"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "batch": batch_id,
        "suite": suite.name,
        "winner": winner.label,
        "protocol": winner.protocol,
        "results": {r.label: f"{r.passes}/{r.runs}" for r in ranked},
    }
    if winner.enable_thinking is not None:
        updates["tuned"]["enable_thinking"] = winner.enable_thinking
    return name, updates


def tune(
    base: Config,
    protocols: list[str],
    store: Store,
    *,
    suite: Suite | None = None,
    repeat: int = 2,
    thinking: bool = False,
    task_ids: list[str] | None = None,
    reporter_factory: Any = None,
    progress: Any = None,
) -> tuple[str, list[Ranked], Suite]:
    suite = suite or builtin_suite()
    vs = variants(base, protocols, thinking)
    runner = SuiteRunner(
        suite,
        [(label, cfg) for label, cfg, _, _ in vs],
        store,
        repeat=repeat,
        task_ids=task_ids,
        oracle=False,
        reporter_factory=reporter_factory,
        progress=progress,
        kind="tune",
    )
    batch_id, _ = runner.run()
    meta = {label: (proto, think) for label, _, proto, think in vs}
    return batch_id, rank(summarize(store, batch_id), meta), suite


def save(cfg: Config, ranked: list[Ranked], batch_id: str, suite: Suite) -> tuple[Path, str]:
    name, updates = profile_updates(cfg, ranked[0], ranked, batch_id, suite)
    return save_user_profile(name, updates), name
