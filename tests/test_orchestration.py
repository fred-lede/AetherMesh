from __future__ import annotations

import asyncio

import pytest

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


def test_memory_wiring_streaming(monkeypatch) -> None:
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
        result = list(service.handle_streaming_chat(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        ).iterator)
    except Exception:
        pass
    assert len(records) > 0, "memory_manager.episodic.record was never called in streaming path"
    assert records[-1]["success"] is False


class _FakeAdapter:
    def __init__(self, chat_result=None, stream_result=None):
        self._chat = chat_result
        self._stream = stream_result

    def is_available(self) -> bool:
        return True

    def chat(self, payload):
        return self._chat

    def stream(self, payload):
        return iter(self._stream or [])

    def responses(self, payload):
        return self._chat


def _make_router(monkeypatch, provider: str) -> "RouterService":
    from runtime.orchestration.openai_handler import RouterService
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    service = RouterService()
    service.registry = {
        "models": [
            {"name": "gpt-4o", "provider": provider, "capabilities": ["chat", "responses"]}
        ]
    }
    monkeypatch.setattr(service, "_resolve_provider_and_worker", lambda payload, allow_queue=False: (provider, {}))
    monkeypatch.setattr(service, "_normalize_payload_for_provider", lambda p, provider, worker: p)
    return service


@pytest.mark.parametrize("provider", ["openai", "ollama", "gemini", "nvidia_nim", "ollama_cloud"])
def test_token_usage_wiring_chat(monkeypatch, provider) -> None:
    captured: list[dict] = []
    service = _make_router(monkeypatch, provider)
    monkeypatch.setattr(
        "runtime.orchestration.openai_handler.RouterService._record_token_usage",
        lambda self, user_id, api_key_id, input_tokens, output_tokens, provider, model: captured.append(
            {"user_id": user_id, "api_key_id": api_key_id, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "provider": provider, "model": model}
        ),
    )
    monkeypatch.setattr(
        service, "_adapter",
        lambda provider, worker: _FakeAdapter(chat_result={
            "usage": {"prompt_tokens": 11, "completion_tokens": 22},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }),
    )
    service.handle_chat({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, user_id=1, api_key_id=2)
    assert captured, "record_token_usage never called for chat path"
    last = captured[-1]
    assert last["provider"] == provider
    assert last["model"] == "gpt-4o"
    assert last["user_id"] == 1
    assert last["api_key_id"] == 2
    assert last["input_tokens"] == 11
    assert last["output_tokens"] == 22


@pytest.mark.parametrize("provider", ["openai", "ollama", "ollama_cloud"])
def test_token_usage_wiring_streaming(monkeypatch, provider) -> None:
    captured: list[dict] = []
    service = _make_router(monkeypatch, provider)
    monkeypatch.setattr(
        "runtime.orchestration.openai_handler.RouterService._record_token_usage",
        lambda self, user_id, api_key_id, input_tokens, output_tokens, provider, model: captured.append(
            {"user_id": user_id, "api_key_id": api_key_id, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "provider": provider, "model": model}
        ),
    )
    monkeypatch.setattr(
        service, "_adapter",
        lambda provider, worker: _FakeAdapter(stream_result=[
            {"usage": {"prompt_tokens": 33, "completion_tokens": 44},
             "choices": [{"delta": {"content": "x"}}]},
        ]),
    )
    list(service.handle_streaming_chat(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        user_id=1, api_key_id=2,
    ).iterator)
    assert captured, "record_token_usage never called for streaming path"
    last = captured[-1]
    assert last["provider"] == provider
    assert last["user_id"] == 1
    assert last["input_tokens"] == 33
    assert last["output_tokens"] == 44


def test_token_usage_skipped_without_user(monkeypatch) -> None:
    from runtime.orchestration.openai_handler import RouterService
    called: list[tuple] = []
    monkeypatch.setattr(
        "runtime.orchestration.openai_handler.record_token_usage",
        lambda db, **kw: called.append(tuple(kw.items())),
    )
    service = RouterService()
    service._record_token_usage(None, None, input_tokens=10, output_tokens=20, provider="openai", model="gpt-4o")
    assert called == [], "record_token_usage should be skipped when user_id is None"
    service._record_token_usage(1, 2, input_tokens=10, output_tokens=20, provider="openai", model="gpt-4o")
    assert len(called) == 1
    kwargs = dict(called[0])
    assert kwargs["user_id"] == 1
    assert kwargs["api_key_id"] == 2
    assert kwargs["input_tokens"] == 10
    assert kwargs["output_tokens"] == 20


def test_execution_plan_wraps_graph() -> None:
    g = ExecutionGraph()
    g.add_node(ExecutionNode(id="a", node_type=NodeType.LLM_CALL))
    ep = ExecutionPlan(graph=g, task="test", context={})
    assert ep.task == "test"
    assert ep.node_count == 1
