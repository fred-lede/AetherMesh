# Wiring Infrastructure + AgentLoop Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire memory recording into OpenAI chat/responses paths and make AgentLoop a functional DAG executor with registered handlers.

**Architecture:** Add ~30 lines to openai_handler.py for `memory_manager.episodic.record()` calls following the Anthropic endpoint pattern. Rewrite agent_loop.py (~130 lines) to register 4 handler types (LLM_CALL, TOOL_CALL, CONDITIONAL, AGENT_CALL) so GraphExecutor can execute real work. Add tests for both.

**Tech Stack:** Python 3.11, asyncio, pytest, in-memory stores

---

### Task 1: Add memory recording to handle_chat (non-streaming)

**Files:**
- Modify: `runtime/orchestration/openai_handler.py`
- Test: `tests/test_orchestration.py` (next task)

- [ ] **Step 1: Add import for memory_manager**

Add after line 28 in openai_handler.py:
```python
from runtime.memory.memory_manager import memory_manager
```

- [ ] **Step 2: Add success record before `return response` in handle_chat**

After line 136 (`return response`), but before the actual return — add recording before line 136:
```python
            if response:
                usage = response.get("usage") or {}
                memory_manager.episodic.record(
                    model=effective_payload.get("model", ""),
                    provider=provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=True,
                    token_count={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                )
```

Actual edit: insert before `return response` at line 136.

- [ ] **Step 3: Add success record in fallback success path**

Before `return adapter_instance.chat(effective_payload)` at line 159:
```python
                fallback_response = adapter_instance.chat(effective_payload)
                memory_manager.episodic.record(
                    model=effective_payload.get("model", ""),
                    provider=provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=True,
                )
                return fallback_response
```

Note: The existing code has `return adapter_instance.chat(effective_payload)` which calls chat() then returns. We need to split into two lines to capture the response and record before returning.

- [ ] **Step 4: Add error record before primary error raises**

Before `raise self._provider_http_error(exc, code=error_code) from exc` at line 164 (and the similar raise at line 192 and 196):
```python
                memory_manager.episodic.record(
                    model=effective_payload.get("model", ""),
                    provider=provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    success=False,
                    error=str(exc)[:200],
                )
```

Three places: line 163-164 (fallback error), line 191 (fallback timeout error), line 196 (non-fallback error).

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "feat: add memory recording to handle_chat"
```

### Task 2: Add memory recording to handle_streaming_chat

**Files:**
- Modify: `runtime/orchestration/openai_handler.py`

- [ ] **Step 1: Add success record in streaming finally block**

In the `finally` block at line 303-323, after `self._finalize_request(...)` call:
```python
                if not error:
                    memory_manager.episodic.record(
                        model=str(state["payload"]).get("model", ""),
                        provider=str(state["provider"]),
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                    )
```

- [ ] **Step 2: Add error record before fallback error yields in streaming**

Before `yield {"error": ...}` at lines 251-257 and 289-296:
```python
            memory_manager.episodic.record(
                model=fallback_payload.get("model", ""),
                provider="ollama",
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(fallback_exc)[:200],
            )
```

- [ ] **Step 3: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "feat: add memory recording to handle_streaming_chat"
```

### Task 3: Add memory wiring tests

**Files:**
- Create/Tests: `tests/test_orchestration.py` (or extend existing)

- [ ] **Step 1: Write test_memory_wiring_chat**

Add to tests/test_orchestration.py:
```python
def test_memory_wiring_chat(monkeypatch):
    records = []
    monkeypatch.setattr(
        "runtime.orchestration.openai_handler.memory_manager.episodic.record",
        lambda **kw: records.append(kw),
    )
    # Create a minimal RouterService and call handle_chat
    from runtime.orchestration.openai_handler import RouterService
    service = RouterService()
    # This should fail (no model specified), but memory should still record the error
    try:
        service.handle_chat({"messages": [{"role": "user", "content": "hi"}]})
    except Exception:
        pass
    assert len(records) > 0
    assert records[-1]["success"] is False
```

- [ ] **Step 2: Run test to verify it fails on current code**

Run: `pytest tests/test_orchestration.py::test_memory_wiring_chat -v`
Expected: The test might pass if the import chain works; adjust based on actual output.

- [ ] **Step 3: Run full test suite to verify**

Run: `python -m pytest tests/ -q --tb=short --ignore=tests/test_dashboard_auth.py -k "not test_openai_resolver_uses_capability_fallback_before_dispatch" 2>&1 | tail -10`
Expected: All tests pass (232+).

- [ ] **Step 4: Commit**

```bash
git add tests/test_orchestration.py
git commit -m "test: add memory wiring tests"
```

### Task 4: Rewrite AgentLoop with handler registration + TOOL/CONDITIONAL/AGENT handlers

**Files:**
- Modify: `runtime/agents/agent_loop.py`

- [ ] **Step 1: Replace agent_loop.py with full implementation**

Write the complete rewrite. The file grows from 49 to ~130 lines.

Imports:
```python
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_result import AgentResult
from runtime.agents.agent_step import AgentStep
from runtime.memory.memory_manager import memory_manager
from runtime.multi_agent.coordinator import coordinator
from runtime.orchestration.execution_plan import ExecutionPlan
from runtime.orchestration.graph import ExecutionNode, NodeType
from runtime.orchestration.graph_executor import GraphExecutor
from runtime.orchestration.planner import Planner
from runtime.orchestration.retry_policy import RetryPolicy
from runtime.tools.tool_executor import ToolExecutor
from runtime.tools.tool_result import ToolCall
```

AgentLoop class:
```python
logger = logging.getLogger("agents.loop")


class AgentLoop:
    def __init__(self) -> None:
        self.executor = GraphExecutor(retry_policy=RetryPolicy(max_retries=2))
        self.planner = Planner()

    async def run(self, context: AgentContext, task: str) -> AgentResult:
        started = time.time()
        result = AgentResult(task=task, started_at=started)
        context.steps = []
        context.task = task

        self.executor.register_handler("llm_call", _make_llm_handler())
        self.executor.register_handler("tool_call", _make_tool_handler())
        self.executor.register_handler("conditional", _make_conditional_handler())
        self.executor.register_handler("agent_call", _make_agent_handler())

        plan = ExecutionPlan.from_task(task, self.planner)
        context.metadata["graph_nodes"] = list(plan.graph.nodes.keys())
        context.metadata["graph_plan"] = "planned"

        exec_result = await self.executor.execute(plan.graph, context={"task": task})

        step = AgentStep(step_number=1, started_at=started)
        step.complete(str(exec_result.output or exec_result.node_results))
        context.add_step(step)

        result.steps = context.steps
        result.output = exec_result.output or exec_result.node_results
        result.finalize()

        memory_manager.episodic.record(
            session_id=context.session_id or task,
            model="agent",
            provider="agent_loop",
            task_summary=task[:200],
            duration_ms=exec_result.elapsed_ms,
            success=exec_result.success,
            error="; ".join(exec_result.node_errors.values()) if exec_result.node_errors else None,
        )

        return result
```

Top-level handler factories (outside class):
```python
def _make_tool_handler():
    executor = ToolExecutor()
    async def handler(node: ExecutionNode) -> Any:
        tool_name = node.config.get("tool", "")
        arguments = node.config.get("arguments", {})
        tool_call = ToolCall(
            id=node.id,
            name=tool_name,
            arguments=arguments,
            source_provider="agent_loop",
            source_model="",
        )
        tool_result = await asyncio.to_thread(executor.execute, tool_call)
        return {"tool": tool_name, "result": tool_result.output, "error": tool_result.is_error}
    return handler


def _make_conditional_handler():
    def _eval(condition: str, ctx: dict) -> bool:
        return bool(eval(condition, {"__builtins__": {}}, ctx))

    async def handler(node: ExecutionNode) -> Any:
        condition = node.config.get("condition", "True")
        ctx = node.config.get("context", {})
        try:
            result = await asyncio.to_thread(_eval, condition, ctx)
        except Exception as exc:
            logger.warning("Conditional eval failed for node %s: %s", node.id, exc)
            result = False
        return result
    return handler


def _make_agent_handler():
    async def handler(node: ExecutionNode) -> Any:
        agent_id = node.config.get("agent_id", "")
        task = node.config.get("task", "")
        sub_result = await coordinator.delegate(task, agent_id)
        return {"agent": agent_id, "success": sub_result.success, "output": str(sub_result.output)}
    return handler
```

- [ ] **Step 2: Run py_compile to verify syntax**

Run: `python -c "import ast; ast.parse(open('runtime/agents/agent_loop.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add runtime/agents/agent_loop.py
git commit -m "feat: rewrite AgentLoop with handler registration + tool/conditional/agent handlers"
```

### Task 5: Add LLM_CALL handler to AgentLoop

**Files:**
- Modify: `runtime/agents/agent_loop.py`

- [ ] **Step 1: Add _make_llm_handler factory function**

Add to agent_loop.py after the existing handler factories:

```python
def _make_llm_handler():
    from runtime.intelligence.execution_selector import execution_selector
    from runtime.orchestration.routing_engine import routing_engine

    async def handler(node: ExecutionNode) -> Any:
        prompt = node.config.get("prompt", "")
        model = node.config.get("model", "default")
        has_tools = bool(node.config.get("tools", False))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        routing_decision = routing_engine.route(
            model=model,
            required_capabilities=["chat"],
        )
        reranked = execution_selector.rerank(
            routing_decision,
            model=model,
            required_capabilities=["chat"],
            has_tools=has_tools,
        )

        provider = reranked.provider
        worker = reranked.worker

        adapter = _adapter_for_provider(provider, worker)
        if adapter is None:
            return {"error": f"No adapter for {provider}/{model}", "text": ""}

        try:
            response = await asyncio.to_thread(adapter.chat, payload)
            text = _extract_text_from_chat(response)
            return {"text": text, "provider": provider, "model": model}
        except Exception as exc:
            logger.error("LLM handler failed for %s/%s: %s", provider, model, exc)
            return {"error": str(exc), "text": ""}

    return handler


def _adapter_for_provider(provider: str, worker: dict[str, Any] | None = None) -> Any:
    from providers.ollama_adapter import OllamaAdapter
    from providers.openai_adapter import OpenAIAdapter
    from providers.gemini_adapter import GeminiAdapter
    from providers.nvidia_nim_adapter import NvidiaNimAdapter
    from providers.ollama_cloud_adapter import OllamaCloudAdapter

    adapters = {
        "ollama": OllamaAdapter,
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
        "nvidia_nim": NvidiaNimAdapter,
        "ollama_cloud": OllamaCloudAdapter,
    }
    cls = adapters.get(provider)
    if cls is None:
        return None
    return cls(worker=worker) if worker else cls()


def _extract_text_from_chat(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    return str(content) if content else ""
```

Add the import for `provider_capability_registry` if needed (for execution_selector compatibility).

- [ ] **Step 2: Run py_compile to verify syntax**

Run: `python -c "import ast; ast.parse(open('runtime/agents/agent_loop.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add runtime/agents/agent_loop.py
git commit -m "feat: add LLM_CALL handler with routing + intelligence scoring"
```

### Task 6: Add AgentLoop tests

**Files:**
- Create: `tests/test_agent_loop.py`

- [ ] **Step 1: Write test_llm_call_handler**

```python
from __future__ import annotations

import asyncio

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_loop import AgentLoop
from runtime.agents.agent_result import AgentResult
from runtime.orchestration.graph import ExecutionGraph, ExecutionNode, NodeType


def _make_text_handler(expected_text: str = "hello response"):
    async def handler(node):
        return {"text": expected_text, "provider": "ollama", "model": "test"}
    return handler


def test_llm_call_handler():
    loop = AgentLoop()
    loop.executor.register_handler("llm_call", _make_text_handler("hello world"))
    graph = ExecutionGraph()
    node = ExecutionNode(id="n1", node_type=NodeType.LLM_CALL, config={"prompt": "say hello"})
    graph.add_node(node)

    result = asyncio.run(loop.executor.execute(graph))
    assert result.success is True
    assert result.node_results["n1"]["text"] == "hello world"
```

- [ ] **Step 2: Write test_tool_call_handler**

```python
def _make_mock_tool_handler():
    async def handler(node):
        return {"tool": node.config.get("tool", ""), "result": "mock tool output", "error": False}
    return handler


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
```

- [ ] **Step 3: Write test_conditional_skip**

```python
def _make_false_handler():
    async def handler(node):
        return False
    return handler


def test_conditional_skip():
    loop = AgentLoop()
    loop.executor.register_handler("conditional", _make_false_handler())

    graph = ExecutionGraph()
    cond = ExecutionNode(id="c1", node_type=NodeType.CONDITIONAL, config={"condition": "False"})
    worker = ExecutionNode(id="w1", node_type=NodeType.LLM_CALL, config={"prompt": "should not run"}, dependencies=["c1"])
    graph.add_node(cond)
    graph.add_node(worker)

    result = asyncio.run(loop.executor.execute(graph))
    # CONDITIONAL returns False, but graph executor doesn't skip dependent nodes automatically
    # This test verifies the handler returns False correctly
    assert result.node_results["c1"] is False
```

Note: GraphExecutor doesn't skip dependent nodes on condition false. The CONDITIONAL node is for data flow — downstream nodes check the result. This is acceptable for the minimal implementation; full conditional branching is future work.

- [ ] **Step 4: Write test_parallel_execution**

```python
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
```

- [ ] **Step 5: Write test_agent_loop_full_run**

```python
def _make_mock_conditional_handler():
    async def handler(node):
        return True
    return handler


async def test_agent_loop_full_run():
    loop = AgentLoop()
    loop.executor.register_handler("llm_call", _make_text_handler("full run response"))
    loop.executor.register_handler("tool_call", _make_mock_tool_handler())
    loop.executor.register_handler("conditional", _make_mock_conditional_handler())

    context = AgentContext(session_id="test-session", task="test task")
    result = await loop.run(context, "say hello")

    assert result.success
    assert result.output is not None
    assert isinstance(result, AgentResult)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_agent_loop.py -v`
Expected: All 5 tests pass.

- [ ] **Step 7: Run full test suite to check regressions**

Run: `python -m pytest tests/ -q --tb=short --ignore=tests/test_dashboard_auth.py -k "not test_openai_resolver_uses_capability_fallback_before_dispatch" 2>&1 | tail -10`
Expected: 237+ pass (232 + 5 new).

- [ ] **Step 8: Commit**

```bash
git add tests/test_agent_loop.py
git commit -m "test: add AgentLoop tests"
```
