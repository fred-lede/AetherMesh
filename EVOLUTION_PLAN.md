# AetherMesh Evolution Plan — Runtime Intelligence

## Current State Summary

| Area | Status | Lines |
|------|--------|-------|
| Orchestration/Routing | Mature | 644+140+69 |
| OpenAI Adapter | Mature | 822 |
| Anthropic Adapter | Mature | 657 |
| Streaming (passthrough) | Mature | 183 |
| Execution Lifecycle | Complete | 81 |
| GPU Runtime + GPU OS | Mature | 193+206 |
| Security | Mature | 252+233 |
| Tools | Mature | 700+ |
| Agents + Multi-Agent | Mature | 153+355 |
| MCP | Good bones | 558 |
| Sessions | Light (in-memory) | 137 |
| Responses Runtime | Light | 50 |
| **Intelligence (Phase 1)** | **Complete** | 154+168 |
| **Memory (Phase 2)** | **Complete** | 356 |
| **Execution Graph (Phase 3)** | **Complete** | 359 |
| **Observability (Phase 5)** | **Complete** | 135 |
| **Tests** | **118 passing** | 7 test files |

---

## Architecture — AIIH-Specific Adaptation

The 10 phases from the vision prompt are reorganized into **6 AIIH phases** based on actual codebase state:

```
PHASE 1: Runtime Intelligence Engine  (wires the scoring layer)
PHASE 2: Memory Runtime               (foundational — everything needs memory)
PHASE 3: Execution Graph Runtime      (DAG orchestration, replaces linear agent loop)
PHASE 4: Multi-Agent + Agent Runtime  (builds on graph + memory)
PHASE 5: Real Streaming + Observability (runtime-native streaming, dashboard evolution)
PHASE 6: GPU OS + Security Hardening  (deep optimization)
```

---

## PHASE 1 — RUNTIME INTELLIGENCE ENGINE

**Target: Replace static routing rules with live scoring**

### Current gap

`ModelRoutingEngine.route()` already does capability scoring against `CAPABILITY_PROVIDER_SCORES` (static dict). But the scores are hardcoded constants, and the intent-based scoring described in the prompt (`context size`, `estimated cost`, `session affinity`, `historical reliability`) does not exist.

### Implementation

**1a. `runtime/intelligence/provider_scoring.py`** — Live scoring layer

```python
class ProviderCapabilityRegistry:
    # Wraps existing capabilities.py + routing_engine.py provider scoring
    # Adds: context_size, estimated_cost, session_affinity, reliability
    
    def register_provider(self, name, capabilities: ProviderCapabilities)
    def unregister_provider(self, name)
    def get_capabilities(self, name) -> ProviderCapabilities | None
    def get_best_provider(self, required: list[str], context: ScoringContext) -> ScoredProvider
    def score_provider(self, name, required: list[str], context: ScoringContext) -> float
```

`ScoringContext` includes:
- `required_capabilities: list[str]`
- `message_count: int` (session depth → context window needs)
- `estimated_input_tokens: int`
- `has_tools: bool`
- `has_vision: bool`
- `has_thinking: bool`
- `session_id: str | None` (for session affinity)
- `affinity_preference: float` (0=no preference, 1=strong affinity)

**1b. `runtime/intelligence/execution_selector.py`** — Decision engine

Wraps the routing engine's scoring to inject live signal:
- Read GPU pressure from `vram_scheduler` (already exists)
- Read warm pool from `warm_pool` (already exists)
- Read model affinity from `model_affinity_tracker` (already exists)
- Read provider health/latency from `routing_engine` (already exists)
- Combine → override `RoutingDecision.score` dynamically

**1c. Wire into existing pipeline**

- `messages_adapter.py:65` — after `routing_engine.route()`, pass through `execution_selector.rerank()` to adjust score based on live intelligence
- The `score < 10` fallback threshold becomes dynamic per-request

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/intelligence/__init__.py` | **New** |
| `runtime/intelligence/provider_scoring.py` | **New** |
| `runtime/intelligence/execution_selector.py` | **New** |
| `runtime/orchestration/routing_engine.py` | **Minor** — expose live signals to new selector |
| `router/anthropic/messages_adapter.py` | **Minor** — wire selector after route() |
| `router/openai/chat_adapter.py` | **Minor** — wire selector after route() |

### Verification

- Existing 37 tests still pass
- New unit tests for `ProviderCapabilityRegistry.score_provider()` with mock contexts
- Manual: curl with `thinking: true` → verify `nvidia_nim` scores higher for `z-ai/glm4.7`

---

## PHASE 2 — MEMORY RUNTIME

**Target: Persistent, semantic, episodic memory for agents**

### Why this order

Memory is foundational. Execution graphs (Phase 3) need memory for state persistence. Multi-agent (Phase 4) needs shared memory. Agents need memory to be useful.

### Implementation

**2a. `runtime/memory/short_term.py`** — Conversation/session memory

Session-scoped, in-memory, auto-expiry. Wraps the existing `SessionStore` but adds:
- Token-count-aware truncation
- Structured message indexing (role, timestamp, tool_call_id)
- `summarize()` — produces a running summary for context window management

**2b. `runtime/memory/semantic_memory.py`** — Vector/semantic retrieval

Abstract interface with:
- `store(embedding_id, text, metadata)`
- `search(query_embedding, top_k) -> list[MemoryHit]`
- Supports pluggable backends (in-memory dict first, ChromaDB/FAISS later)

Initial backend: in-memory with numpy cosine similarity. No external dependencies.

**2c. `runtime/memory/episodic_memory.py`** — Execution history

Records:
- What task was run
- What model/provider was used
- What tools were called
- What errors occurred
- How long it took
- Whether it succeeded

This feeds back into Phase 1's `historical_reliability` scoring signal.

**2d. `runtime/memory/memory_manager.py`** — Unified access

```python
class MemoryManager:
    short_term: ShortTermMemory
    semantic: SemanticMemory  
    episodic: EpisodicMemory
    
    def store_execution(self, ctx: ExecutionContext)
    def search_relevant(self, query: str, session_id: str) -> list[MemoryHit]
```

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/memory/__init__.py` | **New** |
| `runtime/memory/short_term.py` | **New** |
| `runtime/memory/semantic_memory.py` | **New** |
| `runtime/memory/episodic_memory.py` | **New** |
| `runtime/memory/memory_manager.py` | **New** |
| `runtime/orchestration/execution_lifecycle.py` | **Wire** — store execution to episodic memory |
| `config/settings.py` | **Minor** — add `memory_*` settings |

### Verification

- Unit tests: store + search semantic memory with known vectors
- Integration: run a curl, verify episodic memory records it
- Manual: `memory_manager.search_relevant("previous GPU allocation")` returns hits

---

## PHASE 3 — EXECUTION GRAPH RUNTIME

**Target: DAG-based orchestration replacing linear agent loop**

### Current gap

`AgentLoop.run()` is a skeleton. The real "agent loop" lives inside `RouterService.handle_chat()` and `messages_adapter.py` — which is fundamentally linear (request → route → execute → stream → respond).

### Implementation

**3a. `runtime/orchestration/graph.py`** — DAG definition

```python
@dataclass
class ExecutionNode:
    id: str
    type: NodeType  # LLM_CALL, TOOL_CALL, CONDITIONAL, PARALLEL, FAN_OUT, FAN_IN, RETRY
    config: dict
    dependencies: list[str]

@dataclass  
class ExecutionGraph:
    nodes: dict[str, ExecutionNode]
    edges: list[tuple[str, str]]  # (from_node, to_node)
    entry_points: list[str]
    
    def validate(self) -> list[str]  # cycle detection, orphan detection
    def topological_sort(self) -> list[str]
    def parallel_groups(self) -> list[list[str]]
```

**3b. `runtime/orchestration/graph_executor.py`** — Async DAG runner

```python
class GraphExecutor:
    async def execute(self, graph: ExecutionGraph, context: ExecutionContext) -> ExecutionResult:
        # Topological sort
        # For each parallel group: asyncio.gather()
        # Support pause/resume/cancel via context
        # Store execution trace
```

Supports:
- `asyncio.gather()` for parallel branches
- Conditional nodes (skip branch based on previous output)
- Retry nodes (configurable retry policy)
- Streaming execution state updates

**3c. `runtime/orchestration/planner.py`** — Task → Graph

Takes a user request and produces an execution graph:
```python
class Planner:
    def plan(self, task: str, context: ExecutionContext) -> ExecutionGraph
```

Initial implementation: template-based (pattern match → known graph templates):
- Research task → `web_search -> fetch -> summarize -> generate`
- Code task → `retrieve_context -> generate -> test`
- Simple chat → `generate` (single node, passthrough)

The `AgentLoop.run()` skeleton in `runtime/agents/agent_loop.py` becomes a thin wrapper around `Planner.plan()` + `GraphExecutor.execute()`.

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/orchestration/graph.py` | **New** |
| `runtime/orchestration/graph_executor.py` | **New** |
| `runtime/orchestration/planner.py` | **New** |
| `runtime/orchestration/execution_plan.py` | **New** |
| `runtime/orchestration/retry_policy.py` | **New** |
| `runtime/agents/agent_loop.py` | **Rewrite** — wrap Planner + GraphExecutor |
| `runtime/agents/agent_context.py` | **Extend** — add graph, execution state |
| `runtime/agents/agent_step.py` | **Extend** — add graph node tracking |

### Verification

- Unit tests: construct a research graph, execute it with mock nodes
- Integration: `planner.plan("search the web for AI news")` returns a graph with 4+ nodes
- Manual: run through the agent loop with a graph that calls real tools

---

## PHASE 4 — MULTI-AGENT + AGENT RUNTIME

**Target: Planner/worker architecture with shared memory coordination**

### Why this order

Needs the graph runtime (Phase 3) for execution orchestration and memory (Phase 2) for shared state.

### Implementation

**4a. `runtime/multi_agent/coordinator.py`** — Agent orchestration

```python
class Coordinator:
    agents: dict[str, AgentRuntime]
    
    async def delegate(self, task: str, agent_id: str, context: ExecutionContext) -> AgentResult
    async def fan_out(self, task: str, agent_ids: list[str]) -> list[AgentResult]
    async def orchestrate(self, plan: ExecutionGraph) -> list[AgentResult]
```

**4b. `runtime/multi_agent/planner_agent.py`** — Decomposition agent

Receives a complex task, breaks it into subtasks, delegates to worker agents.

**4c. `runtime/multi_agent/worker_agent.py`** — Execution agent

Runs a single subtask with its own tool set and memory context.

**4d. `runtime/multi_agent/shared_memory.py`** — Cross-agent memory

Wraps `semantic_memory` + `episodic_memory` with agent-scoped access controls.

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/multi_agent/__init__.py` | **New** |
| `runtime/multi_agent/coordinator.py` | **New** |
| `runtime/multi_agent/planner_agent.py` | **New** |
| `runtime/multi_agent/worker_agent.py` | **New** |
| `runtime/multi_agent/shared_memory.py` | **New** |
| `runtime/orchestration/graph_executor.py` | **Extend** — support agent node types |

### Verification

- Integration: planner/worker for "write a test for module X" → plan → test sub-agent runs
- Manual: multi-agent research task; verify shared memory passes results between agents

---

## PHASE 5 — REAL STREAMING + OBSERVABILITY

**Target: Runtime-native streaming with structured events**

### Current gap

Streaming is provider passthrough: the router wraps the provider's response stream and converts format (OpenAI SSE → Anthropic SSE). There is no runtime-controlled streaming, no reasoning traces, no graph execution streaming.

### Implementation

**5a. `runtime/streaming/stream_controller.py`** — Runtime-controlled stream

```python
class StreamController:
    def __init__(self):
        self._events: asyncio.Queue[StreamEvent]
    
    async def push(self, event: StreamEvent)
    async def __aiter__(self) -> AsyncIterator[StreamEvent]
    def cancel(self)
```

**5b. `runtime/streaming/stream_state.py`** — Stream lifecycle

```python
@dataclass
class StreamState:
    status: StreamStatus  # ACTIVE, PAUSED, CANCELLED, COMPLETED
    phase: ExecutionPhase
    tokens_generated: int
    tool_calls_pending: list[str]
    errors: list[str]
```

**5c. Structured stream events**

- `reasoning_start / reasoning_delta / reasoning_end` — for thinking traces
- `tool_execution_start / tool_execution_delta / tool_execution_end` — tool call progress
- `graph_node_start / graph_node_complete` — DAG execution updates
- `provider_switch` — when fallback kicks in mid-stream

**5d. Dashboard — Execution graph visualization**

Extend the existing dashboard to show:
- Live graph execution (nodes lighting up as they execute)
- Execution trace viewer
- Provider scoring breakdown per request

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/streaming/__init__.py` | **New** |
| `runtime/streaming/stream_controller.py` | **New** |
| `runtime/streaming/stream_state.py` | **New** |
| `runtime/streaming/reasoning_stream.py` | **New** |
| `runtime/streaming/tool_stream.py` | **New** |
| `runtime/streaming/agent_stream.py` | **New** |
| `runtime/orchestration/streaming.py` | **Extend** — wire StreamController |
| `dashboard/dashboard_server.py` | **Extend** — graph visualization, trace viewer |

### Verification

- Unit: StreamController push/aiter cycle
- Integration: curl with streaming → verify structured events appear
- Manual: open dashboard, verify execution tracing shows real-time node completion

---

## PHASE 6 — GPU OS + SECURITY HARDENING

**Target: Deep infrastructure optimization**

### 6a. GPU Runtime OS

Extend `runtime/gpu/`:

- `runtime/gpu/prediction/load_predictor.py` — Predict upcoming GPU load based on session patterns and active requests
- `runtime/gpu/warm_manager.py` — Hot model pools, predictive loading (extends existing `warm_pool.py`)
- `runtime/gpu/allocation/session_allocator.py` — Session-aware VRAM allocation
- `runtime/gpu/topology/scheduler.py` — Topology-aware scheduling for heterogeneous GPUs (5090, 4070 Ti, P40, M4)

### 6b. Security Hardening

Extend `runtime/security/`:

- `runtime/security/runtime_guard.py` — Per-runtime permission boundaries (tool runtime can't access GPU memory, etc.)
- `runtime/security/execution_limits.py` — Per-session execution budget (max tokens, max tool calls, max wall time)
- `runtime/security/capability_firewall.py` — Provider isolation: "this provider can only be used for chat, not for tool execution"

### What changes vs. what's new

| File | Action |
|------|--------|
| `runtime/gpu/prediction/__init__.py` | **New** |
| `runtime/gpu/prediction/load_predictor.py` | **New** |
| `runtime/gpu/warm_manager.py` | **New** (extends warm_pool.py) |
| `runtime/gpu/allocation/__init__.py` | **New** |
| `runtime/gpu/allocation/session_allocator.py` | **New** |
| `runtime/gpu/topology/__init__.py` | **New** |
| `runtime/gpu/topology/scheduler.py` | **New** |
| `runtime/gpu/vram_scheduler.py` | **Extend** — integrate with prediction + topology |
| `runtime/security/runtime_guard.py` | **New** |
| `runtime/security/execution_limits.py` | **New** |
| `runtime/security/capability_firewall.py` | **New** |

### Verification

- Unit: load predictor with mock request queue
- Integration: allocator with known VRAM profiles
- Manual: verify capability firewall blocks unauthorized provider usage

---

## Summary — File Change Count

| Phase | New Files | Changed Files | Total |
|-------|-----------|---------------|-------|
| 1 — Intelligence | 3 | 3 | 6 |
| 2 — Memory | 5 | 2 | 7 |
| 3 — Execution Graph | 5 | 3 | 8 |
| 4 — Multi-Agent | 5 | 1 | 6 |
| 5 — Streaming + Observability | 5 | 2 | 7 |
| 6 — GPU OS + Security | 10 | 2 | 12 |
| **Total** | **33** | **13** | **46** |

## Status — All 6 Phases Complete

All phases have been implemented, tested (118 tests), wired into both routers, and integrated into the dashboard. See `README.md` for the current state.

## Architectural Rules (from the original prompt, AIIH-adapted)

- `router/` stays protocol-only (FastAPI routes + format conversion)
- `runtime/` is execution-only (no FastAPI, no format conversion)
- `providers/` stays provider-only (API wrappers)
- Intelligence layer composes existing signals (GPU pressure, warm pool, health, latency) — does not duplicate them
- Memory is generic (provider-agnostic, model-agnostic)
- Tools stay provider-agnostic (no OpenAI-specific tool format in runtime/)
- Each phase is independently deployable. No phase requires the next.
