# Wiring Infrastructure + AgentLoop Rewrite

**Date**: 2026-05-25
**Status**: Design Approved
**Area**: Runtime Infrastructure / Agent Execution

## Problem

After 23 phases of refactoring and evolution, AetherMesh has 120+ runtime modules with mature implementations for intelligence scoring, memory, multi-agent coordination, graph execution, and tool execution. However, several critical gaps prevent these modules from being used in production request paths:

1. **Memory** (`runtime/memory/`) is wired into the Anthropic endpoint (`messages_adapter.py`) but NOT into the OpenAI chat or Responses paths
2. **Intelligence** (`runtime/intelligence/`) scoring is wired into Anthropic's `routing_engine.route()` path but not into AgentLoop or the OpenAI dispatch
3. **AgentLoop** (`runtime/agents/agent_loop.py`) is a 49-line skeleton that never calls `register_handler()`, so `GraphExecutor` executes DAGs without any actual LLM/tool handlers
4. AgentLoop has zero tests

## Scope

Minimal wiring approach — add missing connections without refactoring existing dispatch mechanisms.

| Subsystem | Files Changed | Action |
|---|---|---|
| Memory | `runtime/orchestration/openai_handler.py` | Add `memory_manager.episodic.record()` to handle_chat + handle_streaming_chat |
| Intelligence | `runtime/agents/agent_loop.py` | Wire execution_selector.rerank() into LLM_CALL handler |
| AgentLoop | `runtime/agents/agent_loop.py` | Register 4 handlers, make run() actually execute multi-step DAGs |
| Tests | `tests/test_agent_loop.py`, `tests/test_orchestration.py` | New tests for AgentLoop + memory wiring |

NOT changing:
- OpenAI handler's dispatch mechanism (`_resolve_provider_and_worker` stays as-is)
- Anthropic endpoint (already wired)
- GraphExecutor, Planner, Graph models (stable)
- multi_agent routes (separate concern)

## Design

### 1. Memory Wiring (openai_handler.py)

Add `memory_manager.episodic.record()` calls following the same pattern as `messages_adapter.py:238-270`:

```
handle_chat(model, provider, duration_ms, success=True, token_count=usage)
    ↓
handle_chat error path (model, provider, duration_ms, success=False, error=str(exc))
    ↓
handle_streaming_chat completion (model, provider, duration_ms, success=True)
    ↓
handle_streaming_chat fallback / error (model, provider, duration_ms, success=False)
```

Three injection points, ~30 lines total.

### 2. AgentLoop Rewrite (agent_loop.py)

Current state:
- 49 lines, calls `GraphExecutor.execute()` with zero registered handlers
- Single `AgentStep` regardless of DAG complexity
- No memory recording

After rewrite (~130 lines):

```python
class AgentLoop:
    def __init__(self):
        self.executor = GraphExecutor(retry_policy=RetryPolicy(max_retries=2))
        self.planner = Planner()
        self._registry = provider_capability_registry  # for LLM routing

    async def run(self, context, task):
        # Register handlers before execution
        self.executor.register_handler("llm_call", self._llm_handler(context))
        self.executor.register_handler("tool_call", self._tool_handler())
        self.executor.register_handler("conditional", self._conditional_handler())
        self.executor.register_handler("agent_call", self._agent_handler())

        # Plan and execute
        plan = self.planner.plan(task)
        exec_result = await self.executor.execute(plan.graph)

        # Record memory
        memory_manager.episodic.record(
            session_id=context.session_id,
            model="agent",
            provider="agent_loop",
            duration_ms=exec_result.elapsed_ms,
            success=exec_result.success,
        )

        return AgentResult(task=task, output=exec_result.output, ...)
```

#### Handler Implementations

| Handler | Implementation | Dependencies |
|---|---|---|
| LLM_CALL | Route via `routing_engine.route()` + `execution_selector.rerank()` → call provider adapter directly → return response text | routing_engine, execution_selector, adapter factory |
| TOOL_CALL | `tool_executor.execute(ToolCall(name=config["tool"], arguments=config.get("arguments", {})))` | ToolExecutor singleton |
| CONDITIONAL | Evaluate `config["condition"]` against context dict | None |
| AGENT_CALL | `coordinator.delegate(config["agent_id"], config.get("task", ""))` | Coordinator singleton |

**LLM_CALL handler detail** — Does NOT use `RouterService.handle_chat()` (which has its own dispatch + fallback logic). Instead, it does:
1. Resolve `input_from` in config: read previous node results from `exec_result.node_results`
2. Construct prompt: `config["prompt"]` + resolved input from dependencies
3. Build a minimal payload: `{"model": config.get("model", "default"), "messages": [{"role": "user", "content": prompt}], "stream": False}`
4. Call `routing_engine.route()` with required capabilities from config
5. Call `execution_selector.rerank()` on the routing decision
6. Get provider adapter via `_adapter(provider, worker)`
7. Call `adapter.chat(payload)` 
8. Extract and return text content from response
9. On error, log and return error message (no fallback — that's the caller's responsibility)

### 3. No Changes To

- `runtime/orchestration/graph.py` — ExecutionNode, ExecutionGraph, NodeType, NodeStatus stable
- `runtime/orchestration/graph_executor.py` — GraphExecutor, GraphExecutionResult stable
- `runtime/orchestration/planner.py` — Planner templates stable
- `runtime/multi_agent/coordinator.py` — Orchestration stable
- `router/openai/chat_adapter.py` — Thin wrapper stays thin
- `router/anthropic/messages_adapter.py` — Already wired

## Testing

### New: tests/test_agent_loop.py (~120 lines)

| Test | Description |
|---|---|
| `test_llm_call_handler` | Mock LLM handler, single-node graph, verify output |
| `test_tool_call_handler` | Mock tool handler, single-node graph, verify result |
| `test_conditional_skip` | CONDITIONAL node returns False, dependent node skipped |
| `test_parallel_execution` | Two LLM_CALL nodes without dependencies execute in parallel |
| `test_agent_loop_full_run` | Planner.plan() → AgentLoop.run() with all 4 handlers registered |

### Extended: tests/test_orchestration.py (~40 lines)

| Test | Description |
|---|---|
| `test_memory_wiring_chat` | Mock memory_manager, verify handle_chat calls episodic.record() |
| `test_memory_wiring_streaming` | Mock memory_manager, verify handle_streaming_chat records on completion |

## Risks

| Risk | Mitigation |
|---|---|
| AgentLoop registered handlers conflict with existing GraphExecutor usage | AgentLoop creates its own GraphExecutor instance, no shared state |
| Memory wiring adds latency to hot path | episodic.record() is synchronous + lightweight; no I/O in production path |
| Handler factories capture stale references | Handlers created fresh per `run()` call |
| Planner graphs don't match real LLM/tool behavior | Planner templates are simple patterns; real usage will drive template evolution |
