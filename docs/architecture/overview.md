# AetherMesh Runtime Platform — Architecture Overview

## What is AetherMesh?

AetherMesh is a **local-first AI Runtime Mesh Platform** designed for multi-provider,
multi-GPU, and agent-oriented AI systems. It provides a unified runtime for executing
AI workloads across local and cloud models, with built-in tool execution, MCP gateway,
agent orchestration, and GPU-aware scheduling.

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                   Client Apps                        │
│  Claude Code  OpenCode  Cursor  Cline  Chatbox      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                Protocol Adapters (protocols/)        │
│         OpenAI  •  Anthropic  •  MCP  •  Responses  │
└──────────────────────┬──────────────────────────────┘
                       │
 ┌──────────────────────▼──────────────────────────────┐
│              Runtime Engine (runtime/)               │
│  ┌──────┬──────┬─────┬──────┬──────┬──────┬──────┐  │
│  │Tools │Agents│ MCP │Sess. │Resp. │ GPU  │ Sec. │  │
│  └──────┴──────┴─────┴──────┴──────┴──────┴──────┘  │
│  ┌──────┬──────┬────────────┬────────┬───────────┐   │
│  │ Intel│Memory│Multi-Agent │Observ. │  GPU OS   │   │
│  └──────┴──────┴────────────┴────────┴───────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │          Orchestration / Routing              │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Provider Adapters (providers/)          │
│  Ollama  OpenAI  Gemini  NVIDIA NIM  Ollama Cloud   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           GPU Runtime / Control Plane                │
│   RTX 5090  •  RTX 4070 Ti  •  Tesla P40  •  M4    │
└─────────────────────────────────────────────────────┘
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
- **Observability**: Real-time event bus for graph lifecycle, distributed tracing, and metrics collection.
- **GPU OS**: VRAM-aware device pool management, model scheduler with LRU eviction.
- **Security**: Token bucket rate limiter, input validation, API key authentication middleware.
- **GPU Runtime**: VRAM-aware scheduling, model affinity, warm pools.
- **Security Layer**: Tool sandbox, prompt firewall, secret detection, audit logging.

## Directory Layout

```
AetherMesh/
  runtime/          Core execution engine
    intelligence/   Phase 1 — Provider capability scoring + execution selector
    memory/         Phase 2 — Short-term, semantic, episodic memory
    orchestration/  Phase 3 — DAG graphs, executor, planner, retry policy
    multi_agent/    Phase 4 — Coordinator, planner agent, worker agent
    observability/  Phase 5 — Event bus, tracing, metrics collector
    gpu_os/         Phase 6 — GPU device manager, model scheduler
    security/       Phase 6 — Rate limiter, input validator, API key auth
    tools/          Tool runtime + builtin tools
    agents/         Agent orchestration
    mcp/            MCP gateway
    sessions/       Session management
    responses/      Responses API runtime
    gpu/            GPU scheduling
    orchestration/  Routing + lifecycle
  protocols/        Protocol adapters
    openai/
    anthropic/
    mcp/
  providers/        LLM provider adapters + capability registry
  router/           Protocol adapters (openai/ anthropic/ mcp/)
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
