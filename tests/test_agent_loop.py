from __future__ import annotations

import asyncio

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_loop import AgentLoop
from runtime.agents.agent_result import AgentResult
from runtime.orchestration.graph import ExecutionGraph, ExecutionNode, NodeType


# ── Test handler factories ──────────────────────────────────────────────

def _make_text_handler(expected_text: str = "hello response"):
    async def handler(node):
        return {"text": expected_text, "provider": "ollama", "model": "test"}
    return handler


def _make_mock_tool_handler():
    async def handler(node):
        return {"tool": node.config.get("tool", ""), "result": "mock tool output", "error": False}
    return handler


def _make_false_handler():
    async def handler(node):
        return False
    return handler


def _make_mock_conditional_handler():
    async def handler(node):
        return True
    return handler


# ── Tests ───────────────────────────────────────────────────────────────

def test_llm_call_handler():
    loop = AgentLoop()
    loop.executor.register_handler("llm_call", _make_text_handler("hello world"))
    graph = ExecutionGraph()
    node = ExecutionNode(id="n1", node_type=NodeType.LLM_CALL, config={"prompt": "say hello"})
    graph.add_node(node)

    result = asyncio.run(loop.executor.execute(graph))
    assert result.success is True
    assert result.node_results["n1"]["text"] == "hello world"


def test_tool_call_handler():
    loop = AgentLoop()
    loop.executor.register_handler("tool_call", _make_mock_tool_handler())
    graph = ExecutionGraph()
    node = ExecutionNode(id="n1", node_type=NodeType.TOOL_CALL, config={"tool": "web_search"})
    graph.add_node(node)

    result = asyncio.run(loop.executor.execute(graph))
    assert result.success is True
    assert result.node_results["n1"]["tool"] == "web_search"
    assert result.node_results["n1"]["result"] == "mock tool output"


def test_conditional_returns_false():
    loop = AgentLoop()
    loop.executor.register_handler("conditional", _make_false_handler())
    loop.executor.register_handler("llm_call", _make_text_handler("should not run"))

    graph = ExecutionGraph()
    cond = ExecutionNode(id="c1", node_type=NodeType.CONDITIONAL, config={"condition": "False"})
    worker = ExecutionNode(id="w1", node_type=NodeType.LLM_CALL, config={"prompt": "should not run"}, dependencies=["c1"])
    graph.add_node(cond)
    graph.add_node(worker)

    result = asyncio.run(loop.executor.execute(graph))
    # GraphExecutor doesn't auto-skip dependents on condition false.
    # This test verifies the handler returns False correctly.
    assert result.node_results["c1"] is False


def test_parallel_execution():
    loop = AgentLoop()
    loop.executor.register_handler("llm_call", _make_text_handler("parallel"))

    graph = ExecutionGraph()
    a = ExecutionNode(id="a", node_type=NodeType.LLM_CALL, config={"prompt": "task a"})
    b = ExecutionNode(id="b", node_type=NodeType.LLM_CALL, config={"prompt": "task b"})
    graph.add_node(a)
    graph.add_node(b)

    result = asyncio.run(loop.executor.execute(graph))
    assert result.success is True
    assert result.node_results["a"]["text"] == "parallel"
    assert result.node_results["b"]["text"] == "parallel"


async def test_agent_loop_full_run():
    import runtime.agents.agent_loop as agent_loop_mod

    # Monkeypatch all 4 handler factories to return mocks
    # Wrap in lambda so run() can call them to get handlers
    agent_loop_mod._make_llm_handler = lambda: _make_text_handler("full run response")
    agent_loop_mod._make_tool_handler = lambda: _make_mock_tool_handler()
    agent_loop_mod._make_conditional_handler = lambda: _make_mock_conditional_handler()
    agent_loop_mod._make_agent_handler = lambda: _make_mock_conditional_handler()

    loop = AgentLoop()
    context = AgentContext(session_id="test-session", task="test task")
    result = await loop.run(context, "say hello")

    assert result.succeeded
    assert result.output is not None
    assert isinstance(result, AgentResult)
