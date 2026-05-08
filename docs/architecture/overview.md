# AetherMesh AI Runtime Kernel — Architecture Overview

## What is AetherMesh?

AetherMesh is a **local-first AI Runtime Operating System Kernel** designed for multi-provider,
multi-GPU, and agent-oriented AI systems. It provides a unified runtime for executing
AI workloads across local and cloud models, with built-in tool execution, MCP gateway,
agent orchestration, GPU-aware scheduling, and deterministic execution semantics.

## Architecture Layers

```
┌──────────────────────────────────────────────────────────┐
│                       Client Apps                         │
│     Claude Code  OpenCode  Cursor  Cline  Chatbox        │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                Protocol Adapters (router/)                 │
│         OpenAI  •  Anthropic  •  MCP  •  Responses        │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                 AI Runtime Kernel (runtime/)              │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │            Kernel Core (kernel.py)                │    │
│  │  AetherKernel — execution lifecycle orchestrator │    │
│  │  RuntimeLifecycleManager — init/start/pause/...│    │
│  │  EventBus Bridge — graph_event_bus ↔ runtime_event_bus│
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────┬──────┬─────┬──────┬──────┬──────┬──────┐      │
│  │Tools │Agents│ MCP │Sess. │Resp. │ GPU  │ Sec. │      │
│  └──────┴──────┴─────┴──────┴──────┴──────┴──────┘      │
│  ┌──────┬──────┬────────────┬────────┬───────────┐       │
│  │ Intel│Memory│Multi-Agent │Observ. │  GPU OS   │       │
│  └──────┴──────┴────────────┴────────┴───────────┘       │
│  ┌──────────────────────────────────────────────────┐    │
│  │       Kernel Infrastructure (Phases 7-8)          │    │
│  │  context/  events/  state/  replay/  abi/        │    │
│  │  ExecutionCtx  EventBus  StateMachine  Replay     │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Orchestration / Routing                  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│              Provider Adapters (providers/)                │
│  Ollama  OpenAI  Gemini  NVIDIA NIM  Ollama Cloud        │
│  (All adapters support chat + responses + stream)        │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│           GPU Runtime / Control Plane                     │
│   RTX 5090  •  RTX 4070 Ti  •  Tesla P40  •  M4         │
└──────────────────────────────────────────────────────────┘
```

## Key Concepts

- **Runtime-centric**: All execution logic lives in `runtime/`. Protocol adapters (`protocols/`)
  and `router/` only handle format conversion.
- **Tool Runtime**: Models request tool usage; the runtime executes tools and injects results.
- **MCP Gateway**: AetherMesh proxies MCP connections, providing auth, sandboxing, and bridging.
- **Agent Runtime**: Multi-step execution with planner/executor patterns.
- **Intelligence**: Live per-provider scoring with capability matching, context window fit, cost, reliability, session affinity.
- **Memory**: Three-tier memory (short-term, semantic keyword vector search, episodic execution history).
- **Orchestration**: DAG-based execution graphs with parallel group execution, retry policies, and cycle detection.
- **Multi-Agent**: Coordinator-driven delegate, fan-out, and sequential orchestration with shared memory.
- **Observability**: Real-time event bus for graph lifecycle, distributed tracing, and metrics collection. Extended with runtime metrics, execution trace, event/state/replay metric collectors.
- **GPU OS**: VRAM-aware device pool management, model scheduler with LRU eviction.
- **Security**: Token bucket rate limiter, input validation, API key authentication middleware.
- **GPU Runtime**: VRAM-aware scheduling, model affinity, warm pools.
- **Security Layer**: Tool sandbox, prompt firewall, secret detection, audit logging.
- **Execution Context**: Single `ExecutionContext` with 12 typed sub-context states (provider, tool, gpu, session, stream, memory, security, graph, trace, etc.) — replaces fragmented dicts everywhere.
- **Event Bus**: Typed event bus with 26 event types, pub/sub, history, event tracing. All runtime systems communicate through events instead of direct calls.
- **State Machines**: Deterministic state machines with validated transitions across 5 domains (execution, stream, session, agent, provider).
- **Execution Replay**: Full execution recording and replay engine. Supports event stream, graph execution, tool execution, and provider routing replay.
- **Runtime ABI**: 7 stable plugin interfaces (`RuntimeComponent` with `initialize/start/pause/resume/cancel/shutdown`). `RuntimeLifecycleManager` manages all 6 runtime components.
- **Responses API**: Full OpenAI Responses API format support — native passthrough for OpenAI, auto-conversion for all other providers. Supports `input`, `instructions`, `tools`, streaming, tool calls, response management (GET/DELETE/PATCH).

## Directory Layout

```
AetherMesh/
  runtime/          AI Runtime Kernel (v5.0.0)
    intelligence/   Phase 1 — Provider capability scoring + execution selector
    memory/         Phase 2 — Short-term, semantic, episodic memory
    orchestration/  Phase 3 — DAG graphs, executor, planner, retry policy
    multi_agent/    Phase 4 — Coordinator, planner agent, worker agent
    observability/  Phase 5 — Event bus, tracing, metrics
                    + runtime metrics, execution trace, event/state/replay metrics
    gpu_os/         Phase 6 — GPU device manager, model scheduler
    security/       Phase 6 — Rate limiter, input validator, API key auth
    context/        Phase 7 — Unified ExecutionContext (12 sub-context states)
    events/         Phase 7 — Typed event bus (26 event types)
    state/          Phase 7 — State machines (5 domains, validated transitions)
    replay/         Phase 8 — Execution recording + replay engine
    abi/            Phase 8 — Plugin interfaces + RuntimeLifecycleManager
    kernel.py       Phase 8 — AetherKernel bootstrapper
    event_bridge.py         Phase 8 — Event bus bridge
    tools/          Tool runtime + builtin tools + lifecycle adapter
    agents/         Agent loop + lifecycle adapter
    mcp/            MCP gateway
    sessions/       Session management + lifecycle adapter
    responses/      Responses API runtime (full OpenAI Responses format)
    gpu/            GPU scheduling + lifecycle adapter
  protocols/        Protocol adapters
    openai/
    anthropic/
    mcp/
  providers/        LLM provider adapters + capability registry
  router/           Protocol adapters (openai/ anthropic/ mcp/ responses/)
  cli/              CLI client
  clients/          Client SDKs (openai_sdk, anthropic_sdk, mcp_sdk)
  control_plane/    Cluster management
  dashboard/        Web UI
  metrics/          Observability (extended metrics)
  cluster/          Cluster services
  node/             Node/worker agents
  ai_queue/         Async task queue
  config/           Configuration
  docs/             Documentation (16 docs)
```
