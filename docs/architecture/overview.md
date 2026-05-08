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
- **GPU Runtime**: VRAM-aware scheduling, model affinity, warm pools.
- **Security Layer**: Tool sandbox, prompt firewall, secret detection, audit logging.

## Directory Layout

```
AetherMesh/
  runtime/          Core execution engine
    tools/          Tool runtime + builtin tools
    agents/         Agent orchestration
    mcp/            MCP gateway
    sessions/       Session management
    responses/      Responses API runtime
    gpu/            GPU scheduling
    security/       Security layer
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
