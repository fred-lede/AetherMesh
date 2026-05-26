from __future__ import annotations

import asyncio

from runtime.orchestration.graph import ExecutionGraph, ExecutionNode, NodeType, NodeStatus
from runtime.orchestration.graph_executor import GraphExecutor
from runtime.orchestration.planner import Planner
from runtime.orchestration.retry_policy import RetryPolicy
from runtime.orchestration.execution_plan import ExecutionPlan


def test_graph_add_node() -> None:
    g = ExecutionGraph()
    n = ExecutionNode(id="a", node_type=NodeType.LLM_CALL, config={"prompt": "hi"})
    g.add_node(n)
    assert "a" in g.nodes
    assert "a" in g.entry_points


def test_graph_validate_no_errors() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL))
    g.add_node(ExecutionNode(id="b", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    assert g.validate() == []


def test_graph_validate_missing_dep() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL, dependencies=["missing"]))
    errors = g.validate()
    assert any("missing" in e for e in errors)


def test_graph_cycle_detection() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL, dependencies=["b"]))
    g.add_node(ExecutionNode(id="b", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    errors = g.validate()
    assert any("Cycle" in e for e in errors)


def test_topological_sort() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL))
    g.add_node(ExecutionNode(id="b", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    g.add_node(ExecutionNode(id="c", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    g.add_node(ExecutionNode(id="d", node_type=NodeType.LLM_CALL, dependencies=["b", "c"]))
    order = g.topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_parallel_groups() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL))
    g.add_node(ExecutionNode(id="b", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    g.add_node(ExecutionNode(id="c", node_type=NodeType.LLM_CALL, dependencies=["a"]))
    groups = g.parallel_groups()
    assert len(groups) == 2
    assert groups[0] == ["a"]
    assert set(groups[1]) == {"b", "c"}


async def test_graph_executor_runs_simple_graph() -> None:
    exe = GraphExecutor()
    exe.register_handler("llm_call", lambda n: f"processed:{n.config.get('prompt','')}")
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL, config={"prompt": "hello"}))
    r = await exe.execute(g)
    assert r.success is True
    assert r.output == "processed:hello"


async def test_graph_executor_validation_error() -> None:
    exe = GraphExecutor()
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL, dependencies=["missing"]))
    r = await exe.execute(g)
    assert r.success is False
    assert "_graph" in r.node_errors


async def test_retry_policy_success() -> None:
    rp = RetryPolicy(max_retries=2)
    result = await rp.execute(lambda: "ok", node_id="test")
    assert result == "ok"


async def test_retry_policy_failure() -> None:
    rp = RetryPolicy(max_retries=1)
    calls = 0
    async def fail():
        nonlocal calls
        calls += 1
        raise ValueError("boom")
    try:
        await rp.execute(fail, node_id="test")
        assert False, "should have raised"
    except ValueError:
        assert calls == 2  # initial + 1 retry


async def test_retry_policy_exponential_backoff() -> None:
    rp = RetryPolicy(max_retries=2, backoff_base_s=0.01)
    calls = 0
    async def fail():
        nonlocal calls
        calls += 1
        raise ValueError("nope")
    try:
        await rp.execute(fail, node_id="test")
    except ValueError:
        assert calls == 3


def test_planner_research_graph() -> None:
    p = Planner()
    g = p.plan("search for AI news")
    assert len(g.nodes) >= 3
    node_types = {n.node_type.value for n in g.nodes.values()}
    assert "tool_call" in node_types
    assert "llm_call" in node_types


def test_planner_simple_graph() -> None:
    p = Planner()
    g = p.plan("hello")
    assert len(g.nodes) == 1
    assert list(g.nodes.values())[0].node_type == NodeType.LLM_CALL


def test_memory_wiring_chat(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records: list[dict] = []
    monkeypatch.setattr(
        "runtime.orchestration.openai_handler.memory_manager.episodic.record",
        lambda **kw: records.append(kw),
    )
    from runtime.orchestration.openai_handler import RouterService
    service = RouterService()
    service.registry = {
        "models": [
            {
                "name": "gpt-4o",
                "provider": "openai",
                "capabilities": ["chat"],
            }
        ]
    }
    try:
        service.handle_chat({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    except Exception:
        pass
    assert len(records) > 0, "memory_manager.episodic.record was never called"
    assert records[-1]["success"] is False


def test_execution_plan_wraps_graph() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL))
    ep = ExecutionPlan(graph=g, task="test", context={})
    assert ep.task == "test"
    assert ep.node_count == 1
