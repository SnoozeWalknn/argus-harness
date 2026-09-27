from __future__ import annotations

import json

from argus.agent import Agent
from argus.cli import main
from argus.config import load_config
from argus.mock import final
from argus.store import Store
from argus.tokens import TokenCounter, estimate, format_overhead, measure
from tests.conftest import q


def agent_for(server_url: str, workspace, protocol: str) -> Agent:
    cfg = load_config(
        None,
        [
            f"model.base_url={q(server_url)}",
            f"executor.workdir={q(str(workspace))}",
            f"agent.protocol={q(protocol)}",
        ],
    )
    return Agent(cfg, store=Store(":memory:"))


def test_measure_native_attributes_tools(mock, workspace):
    s = mock([], n_ctx=8192)
    ov = measure(agent_for(s.url, workspace, "native"))
    assert ov.method == "server"
    assert ov.total == ov.system + ov.tools + ov.framing
    assert ov.tools > 0 and set(ov.per_tool) == {
        "read", "edit", "write", "bash", "job", "glob", "grep", "ls", "fetch",
    }  # fmt: skip
    # marginal costs are each positive and roughly add up to the tools field
    assert all(v > 0 for v in ov.per_tool.values())
    assert sum(ov.per_tool.values()) <= ov.tools
    assert ov.parts["protocol"] == 0
    assert ov.context_window == 8192
    assert ov.call_cost > 0


def test_constrained_protocols_cost_less(mock, workspace):
    s = mock([])
    ovs = {p: measure(agent_for(s.url, workspace, p)) for p in ("native", "json_schema", "grammar")}
    assert ovs["json_schema"].tools == 0 and ovs["json_schema"].parts["protocol"] > 0
    assert ovs["grammar"].total < ovs["native"].total / 2
    assert ovs["grammar"].call_cost < ovs["json_schema"].call_cost < ovs["native"].call_cost
    assert ovs["grammar"].constraint_chars > 0
    table = format_overhead(list(ovs.items()))
    assert "TOTAL overhead" in table and "grammar" in table


def test_estimate_fallback(workspace):
    agent = agent_for("http://127.0.0.1:9/v1", workspace, "native")
    agent.llm.retries = 0
    ov = measure(agent, counter=TokenCounter(agent.llm))
    assert ov.method == "estimate" and ov.total > 0
    assert ov.notes and "estimated" in ov.notes[0]
    assert estimate("abcd" * 37) == 40


def test_overhead_recorded_on_run(make_agent):
    agent, _ = make_agent([final()])
    r = agent.run("x")
    row = agent.store.run(r.run_id)
    assert row["overhead_tokens"] > 0
    assert json.loads(row["overhead_json"])["protocol"] == "native"
    assert row["context_window"] == 32768


def test_overhead_cli(mock, workspace, capsys):
    s = mock([])
    rc = main(["overhead", "-w", str(workspace), "-o", f"model.base_url={q(s.url)}", "--all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "native" in out and "json_schema" in out and "grammar" in out
    assert "estimate" not in out  # all three measured by the server
    rc = main(["overhead", "-w", str(workspace), "-o", f"model.base_url={q(s.url)}", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["native"]["method"] == "server"
