# AetherMesh AI Runtime Kernel

Local-first AI Runtime OS Kernel for multi-provider, multi-GPU, and agent-oriented AI systems.
Provides OpenAI/Anthropic-compatible API routing, full OpenAI Responses API, GPU-aware
scheduling, deterministic execution semantics, and execution recording/replay — all
running locally with optional cloud provider fallback.

## Architecture

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
│              AI Runtime Kernel (runtime/)                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │            Kernel Core                            │    │
│  │  AetherKernel — execution lifecycle orchestrator  │    │
│  │  RuntimeLifecycleManager — init/start/pause/...   │    │
│  │  EventBus Bridge — graph ↔ runtime event buses   │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────┬──────┬─────┬──────┬──────┬──────┬──────┐      │
│  │Tools │Agents│ MCP │Sess. │Resp. │ GPU  │ Sec. │      │
│  └──────┴──────┴─────┴──────┴──────┴──────┴──────┘      │
│  ┌──────┬──────┬────────────┬────────┬───────────┐       │
│  │ Intel│Memory│Multi-Agent │Observ. │  GPU OS   │       │
│  └──────┴──────┴────────────┴────────┴───────────┘       │
│  ┌──────────────────────────────────────────────────┐    │
│  │       Kernel Infrastructure (Phases 7-8)          │    │
│  │  context/  events/  state/  replay/  abi/        │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│              Provider Adapters (providers/)                │
│  Ollama  OpenAI  Gemini  NVIDIA NIM  Ollama Cloud        │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┘
│           GPU Workers / Control Plane                     │
│   RTX 5090  •  RTX 4070 Ti  •  Tesla P40  •  M4         │
└──────────────────────────────────────────────────────────┘
```

### Kernel Core

The `runtime/kernel.py` `AetherKernel` class is the system bootstrapper. It manages
execution lifecycle via a unified `ExecutionContext` (12 typed sub-contexts), a typed
event bus (26 event types), deterministic state machines (5 domains with validated
transitions), full execution recording/replay, and 7 stable plugin interfaces.

| Component | File | Purpose |
|-----------|------|---------|
| `AetherKernel` | `runtime/kernel.py` | System bootstrapper, execution lifecycle (create/start/pause/resume/cancel/fail/complete/shutdown) |
| `ExecutionContext` | `runtime/context/` | Unified state carrier: provider, tool, GPU, session, stream, memory, security, graph, trace contexts |
| `EventBus` | `runtime/events/` | Typed pub/sub event bus with 26 event types, history, async + sync dispatch |
| `StateMachine` | `runtime/state/` | Deterministic state machines with validated transitions for execution, stream, session, agent, provider |
| `EventBridge` | `runtime/event_bridge.py` | Bidirectional bridge between graph event bus and runtime event bus |
| `ExecutionRecorder` | `runtime/replay/` | Full execution recording — event stream, snapshots, serialization |
| `ReplayEngine` | `runtime/replay/` | Replay recorded executions with event replay, trace rebuilding, graph replay |
| `RuntimeLifecycleManager` | `runtime/abi/` | Registers/manages `RuntimeComponent` plugins — `initialize/start/pause/resume/cancel/shutdown` |
| `Plugin ABIs` | `runtime/abi/` | 7 stable interfaces: agent, GPU, memory, provider, stream, tool, runtime contract |

### Runtime Modules (Phases 1–8)

| Module | Phase | Description |
|--------|-------|-------------|
| `runtime/intelligence/` | 1 | Live per-provider scoring (capability match, context window, cost, reliability). `ExecutionSelector` reranks routing decisions with warm model bonus, session affinity, and context penalty signals. |
| `runtime/memory/` | 2 | Three-tier memory: `ShortTermMemory` (session-scoped), `SemanticMemory` (TF-IDF keyword vector search), `EpisodicMemory` (execution history). Unified via `MemoryManager`. |
| `runtime/orchestration/` | 3 | DAG execution engine: `ExecutionGraph` with cycle detection, topological sort, parallel groups. `GraphExecutor` async runner. `Planner` converts tasks to graphs. `RetryPolicy` with exponential backoff. |
| `runtime/multi_agent/` | 4 | Agent orchestration: `Coordinator` (delegate/fan-out/orchestrate), `PlannerAgent` (task decomposition), `WorkerAgent` (subtask execution), `SharedMemory` (cross-agent state). |
| `runtime/observability/` | 5 | Real-time event bus (`GraphEvent` lifecycle), `Tracer` (trace/span correlation), `MetricsCollector` (counters/histograms/gauges). Wired into `GraphExecutor`. |
| `runtime/gpu_os/` + `runtime/security/` | 6 | GPU device tracking (VRAM pool, utilization, temperature), model scheduler (LRU eviction). Rate limiter (token bucket), input validator, API key auth middleware. |
| `runtime/context/`, `runtime/events/`, `runtime/state/` | 7 | Kernel stabilization: unified `ExecutionContext` with 12 sub-context states, typed event bus (26 event types), deterministic state machines with validated transitions across 5 domains. |
| `runtime/replay/`, `runtime/abi/`, `runtime/kernel.py` | 8 | Execution recording/replay, 7 stable plugin interfaces, `RuntimeLifecycleManager` with `initialize/start/pause/resume/cancel/shutdown` for all components, `AetherKernel` bootstrapper. |

## Quick Start

### Installation

Requires **Python 3.10+** and (for GPU workers) **NVIDIA drivers + CUDA**.

```bash
# 1. Clone and enter the project
git clone <repo-url> AetherMesh
cd AetherMesh

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to match your setup (at minimum: AIIH_NODE_ID, AIIH_NODE_IP)

# 5. (GPU workers only) Install Ollama
# https://ollama.com/download
# Start Ollama instances on ports 11434, 11435, etc.

# 6. Install & start Redis (required for cluster coordination, async tasks)
#
#   macOS (Homebrew):
#     brew install redis && brew services start redis
#
#   Ubuntu / Debian:
#     sudo apt install redis-server
#     sudo systemctl enable --now redis-server
#
#   Windows WSL (Ubuntu):
#     # Same as Ubuntu above inside WSL
#     # Verify: redis-cli ping → PONG
```

### Profiles

Configuration templates for different machine roles are in `profiles/`:

| Profile | File | Use |
|---------|------|-----|
| Control plane | `profiles/control-plane/.env.example` | Main AIIH host (routers, dashboard, control plane) |
| Worker node | `profiles/worker-node/.env.example` | Remote GPU node (Ollama workers, node/worker agents) |
| Ollama GPU 0 | `profiles/worker-node/ollama-gpu0.env.example` | First GPU worker environment |
| Ollama GPU 1 | `profiles/worker-node/ollama-gpu1.env.example` | Second GPU worker environment |

Copy the profile that matches the machine's role to `.env` and adjust the values.

### Run the Kernel (standalone, no cluster)

```python
from runtime.kernel import AetherKernel

kernel = AetherKernel()
await kernel.initialize()
ctx, sm = await kernel.create_execution(session_id="my-session")
await kernel.start_execution(ctx, sm)
# ... do work ...
await kernel.complete_execution(ctx, sm)
await kernel.shutdown()
```

### Run the Full Cluster

```bash
# All platforms — one terminal, one command (starts 8 services)
python -m runtime.launcher

# Start specific services only
python -m runtime.launcher start control_plane openai_router dashboard

# Show status of all services
python -m runtime.launcher status

# Stop all services
python -m runtime.launcher stop

# Custom log directory
python -m runtime.launcher --log-dir /var/log/aethermesh
```

Each service writes its own log file to `logs/<name>.log`. Debug a specific service:
```bash
tail -f logs/openai_router.log        # watch OpenAI router logs
tail -f logs/control_plane.log        # watch control plane logs
```

The launcher starts 8 services in a single process group:
`control_plane` (9200), `openai_router` (8001), `anthropic_router` (8002),
`dashboard` (9001), `metrics` (9100), `node_agent` (9400), `worker_agent` (9300),
`task_worker`.

### Stop Services

```bash
# Graceful stop — sends SIGTERM to all services, waits up to 5s, then SIGKILL
python -m runtime.launcher stop

# Or press Ctrl+C in the terminal where launcher is running
```
When using boot startup configs (systemd/launchd/Task Scheduler):
```bash
# Ubuntu (control plane)
sudo systemctl stop aiih-launcher

# Ubuntu (worker node)
sudo systemctl stop aiih-worker

# macOS (control plane)
launchctl unload ~/Library/LaunchAgents/com.aiih.launcher.plist

# macOS (worker node)
launchctl unload ~/Library/LaunchAgents/com.aiih.worker.plist

# Windows (Task Scheduler)
schtasks /Change /TN "AetherMesh" /DISABLE
schtasks /Change /TN "AetherMesh-Worker" /DISABLE
# or via GUI: taskschd.msc → Disable the task
```

### Boot Startup

Set up the launcher to start automatically on boot:

**Ubuntu (systemd)**
1. Replace `__ROOT_DIR__` in `systemd/aiih-launcher.service` with the absolute path to this project
2. **Replace `fred` in `User=fred` and `Group=fred` with your actual Linux username** — mismatched user will cause permission errors on logs and prevent proper process management
3. Install and enable:
   ```bash
   sudo cp systemd/aiih-launcher.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable aiih-launcher
   sudo systemctl start aiih-launcher
   ```
4. Check status: `sudo systemctl status aiih-launcher`
5. View logs: `sudo journalctl -u aiih-launcher -f`

**macOS (launchd)**
1. Replace `__ROOT_DIR__` in `launchd/com.aiih.launcher.plist.example` with the absolute path
2. Copy and load:
   ```bash
   cp launchd/com.aiih.launcher.plist.example ~/Library/LaunchAgents/com.aiih.launcher.plist
   launchctl load ~/Library/LaunchAgents/com.aiih.launcher.plist
   ```
3. Unload: `launchctl unload ~/Library/LaunchAgents/com.aiih.launcher.plist`
4. View logs: `tail -f logs/launchd.out.log`

**Windows (Task Scheduler)**
1. Create `scripts/start_launcher.bat`:
   ```batch
   @echo off
   cd /d "C:\path\to\AetherMesh"
   .venv\Scripts\python.exe -m runtime.launcher
   ```
2. Open **Task Scheduler** → Create Task
3. **General**: Run whether user is logged on or not, run with highest privileges
4. **Trigger**: At startup
5. **Action**: Start a program → `C:\path\to\AetherMesh\.venv\Scripts\python.exe` with args `-m runtime.launcher`, start in `C:\path\to\AetherMesh`
6. **Settings**: If task fails, restart every 10 minutes

### Remote Worker Nodes

Add GPU machines to the cluster as worker nodes. Each worker runs only the node/worker agents
and Ollama — no routers, dashboard, or control plane.

**Prerequisites** on the worker machine:
- Python 3.10+, NVIDIA drivers, CUDA, Ollama installed
- Network access to the control plane host (ports 9200, 6379)

**Step 1: Clone and install**
```bash
git clone <repo-url> AetherMesh && cd AetherMesh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Configure `.env`** (use the worker-node profile)
```bash
cp profiles/worker-node/.env.example .env
```

Edit these required values:
```ini
AIIH_CONTROL_URL=http://<CONTROL_PLANE_IP>:9200
AIIH_NODE_ID=node-p40-01          # unique per cluster
AIIH_NODE_IP=<WORKER_LAN_IP>      # worker's LAN address
AIIH_REDIS_URL=redis://<CONTROL_PLANE_IP>:6379/0
```

**Step 3: Start Ollama workers** (one per GPU)

Detailed environment templates for each GPU are in `profiles/worker-node/`:
```bash
# GPU 0 — binds to port 11434, uses CUDA device 0
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS=*
export OLLAMA_CONTEXT_LENGTH=64000
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
ollama serve &

# GPU 1 — binds to port 11435, uses CUDA device 1
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_HOST=0.0.0.0:11435
ollama serve &
```

Or use the env files directly:
```bash
env $(cat profiles/worker-node/ollama-gpu0.env.example | grep -v '^#') ollama serve &
env $(cat profiles/worker-node/ollama-gpu1.env.example | grep -v '^#') ollama serve &
```

**Ollama boot startup** (Ubuntu systemd):
```bash
# Edit systemd/aiih-ollama-gpu0.service and aiih-ollama-gpu1.service
# to match your GPU IDs and paths, then:
sudo cp systemd/aiih-ollama-gpu{0,1}.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aiih-ollama-gpu0 aiih-ollama-gpu1
```

Pull models on each worker:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# Pull any other models you need
```

**Step 4: Start node and worker agents**
```bash
# Starts only node_agent + worker_agent on this machine
python -m runtime.launcher start node_agent worker_agent
```

**Boot startup** (so agents start automatically when the worker reboots):

Worker-node service files are in `systemd/aiih-worker.service` and
`launchd/com.aiih.worker.plist.example`. Replace `__ROOT_DIR__` with the actual path,
and **replace `fred` in `User=fred`/`Group=fred` with your Linux username**:

```bash
# Ubuntu
sudo cp systemd/aiih-worker.service /etc/systemd/system/
sudo systemctl enable aiih-worker
sudo systemctl start aiih-worker

# macOS
cp launchd/com.aiih.worker.plist.example ~/Library/LaunchAgents/com.aiih.worker.plist
launchctl load ~/Library/LaunchAgents/com.aiih.worker.plist

# Windows (Task Scheduler) — run only node_agent + worker_agent
schtasks /Create /SC ONSTART /TN "AetherMesh-Worker" /TR "C:\path\to\.venv\Scripts\python.exe -m runtime.launcher start node_agent worker_agent" /RU SYSTEM /RL HIGHEST
```

**Step 5: Register models on the control plane**

On the control plane host, edit `config/models.yaml` to bind models to the new node:
```yaml
models:
  - name: llama3.2:3b
    worker_bindings:
      - node_id: node-p40-01
        port: 11434
        gpu_id: 0
```

**Verify**: The worker appears in the dashboard within 15 seconds, or check directly:
```bash
curl http://<CONTROL_PLANE_IP>:9200/cluster/nodes
curl http://<CONTROL_PLANE_IP>:9200/cluster/workers
```

### Configure Claude Code / Claude Desktop

```
baseUrl: "http://127.0.0.1:8002/v1"
apiKey: "local-dev-key"
```

## Protocol Adapters

### OpenAI-Compatible API (`router/openai_router.py`, port `8001`)

- `POST /v1/chat/completions` — streaming (`stream=true`) and non-streaming
- `POST /v1/responses` — full OpenAI Responses API format
- `GET /v1/responses/{id}` — fetch stored response
- `DELETE /v1/responses/{id}` — delete stored response
- `PATCH /v1/responses/{id}` — update response metadata
- `GET /v1/models` — list available models
- `POST /v1/embeddings` — text embeddings
- `POST /v1/rerank` — document reranking
- `GET /v1/gpu/status` — GPU devices, VRAM, utilization, temperature
- `POST /v1/gpu/models/load` — load a model to a device
- `POST /v1/gpu/models/unload` — unload a model
- `POST /v1/gpu/devices/register` — register a GPU device
- `GET /v1/agent/status` — registered agents and shared memory keys
- `POST /v1/agent/plan` — PlannerAgent task decomposition
- `POST /v1/agent/execute` — execute a task (single or multi-step)
- `POST /v1/agent/register` — register a worker agent

### Anthropic-Compatible API (`router/anthropic_router.py`, port `8002`)

Full Anthropic Messages API format (`POST /v1/messages`) with streaming via SSE.
Supports:
- **Tools**: `tools` definition, `tool_choice` (auto/any/tool), `tool_result` with `is_error`
- **Extended Thinking**: `thinking` blocks, `budget_tokens`, streaming `thinking_delta`
- **Prompt Caching**: `cache_control` on text, image, tool_result, and document blocks
- **Vision**: `image` blocks with `base64`, `url`, `media_type`, `detail`
- **Documents**: `document` blocks converted to text hints
- **Audio Input**: `audio` / `input_audio` blocks translated for compatible providers
- **Computer Use / Bash**: non-function tools mapped as function wrappers
- **Assistant Prefill**: assistant role with content as prefill hint
- **Parameters**: `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`
- **Error mapping**: rate_limit, invalid_request, overloaded, api_error
- **Headers**: `X-Request-Id`, `Retry-After`, rate limit headers

### Responses API

Full OpenAI Responses API (`/v1/responses`) — native passthrough for OpenAI,
auto-conversion for Ollama, Gemini, NVIDIA NIM, and Ollama Cloud. Supports
`input`, `instructions`, `tools`, streaming, tool calls, and response management
(GET/DELETE/PATCH). All provider adapters implement the `responses` capability.
Responses input accepts message objects, strings, and bare content parts such as
`input_text`; streaming emits OpenAI-compatible `response.output_text.delta`
events and includes the assembled text in both top-level `output_text` and
`response.completed.output` for clients that read the final event instead of
deltas.

### Embeddings API

`POST /v1/embeddings` routes requests with `input` and no `messages` as the
`embeddings` capability, so embeddings-only models such as
`nomic-embed-text-v2-moe:latest` are not treated as chat models or routed through
chat fallback logic. Ollama and Ollama Cloud embedding adapters normalize the
common upstream shapes (`embeddings`, legacy `embedding`, and OpenAI-style
`data[].embedding`) into OpenAI-compatible `data[].embedding` rows.

### MCP Gateway (`runtime/mcp/`)

Proxies MCP connections with auth, sandboxing, and bridging.

### Server Tools

`AIIH_SERVER_TOOL_MODE` controls server-tool ownership:
- `reject` — blocks unsupported upstream server tools
- `local` — AIIH handles `web_search` / `web_fetch` locally
- `passthrough` — leaves server tools to Claude Code / runtime client

### Auto Web Search

`AIIH_WEB_TOOLS_AUTO_SEARCH=true` injects web search results into every request
automatically, even if the client does not send tool definitions. The latest user
message is used as the search query, and results + today's date are prepended as
a system message. Works on both OpenAI (port 8001) and Anthropic (port 8002) routers.

## Provider Adapters

| Provider | Adapter | Capabilities |
|----------|---------|-------------|
| Ollama (local) | `providers/ollama_adapter.py` | chat, stream, responses, embeddings, rerank |
| OpenAI | `providers/openai_adapter.py` | chat, stream, responses |
| Gemini | `providers/gemini_adapter.py` | chat, stream, responses, rerank |
| NVIDIA NIM | `providers/nvidia_nim_adapter.py` | chat, stream, responses, embeddings, rerank |
| Ollama Cloud | `providers/ollama_cloud_adapter.py` | chat, stream, responses, embeddings, rerank |

All adapters follow the `ProviderAdapter` interface in `providers/base.py`.

## Configuration

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AIIH_HOST` | `0.0.0.0` | Bind address |
| `AIIH_ROUTER_PORT` | `8001` | OpenAI-compatible API port |
| `AIIH_ANTHROPIC_PORT` | `8002` | Anthropic-compatible API port |
| `AIIH_DASHBOARD_PORT` | `9001` | Dashboard port |
| `AIIH_METRICS_PORT` | `9100` | Prometheus exporter port |
| `AIIH_CONTROL_PORT` | `9200` | Control plane port |
| `AIIH_REDIS_URL` | — | Redis connection for async queue |
| `AIIH_NODE_ID` | — | Unique node identity |
| `AIIH_MAX_WORKER_QUEUE` | — | Max queued tasks per worker |
| `AIIH_WORKER_ASSIGNMENT_TTL` | `900` | Seconds before unreleased sync worker assignments are reclaimed after router crashes |
| `AIIH_PROVIDER_COOLDOWN` | — | Seconds before retrying failed provider |
| `AIIH_DEBUG_RESPONSES` | `false` | Emit compact `/v1/responses` conversion traces to `logs/openai_router.log` |
| `AIIH_DASHBOARD_AUTH_ENABLED` | `false` | Enable dashboard auth |
| `AIIH_API_KEY` | — | Static API key(s) for router auth (comma-separated for multiple) |
| `AIIH_ADMIN_EMAIL` | — | Admin email for first-run bootstrap (creates initial admin) |
| `AIIH_ADMIN_PASSWORD` | — | Admin password for first-run bootstrap |
| `AIIH_DB_PATH` | `config/aiih.db` | SQLite database path for auth subsystem |
| `AIIH_JWT_SECRET` | dev-only fallback | Secret key for JWT token signing |
| `AIIH_SERVER_TOOL_MODE` | `reject` | Server tool policy (`reject`/`local`/`passthrough`) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `NVIDIA_NIM_API_KEY` | — | NVIDIA NIM API key |
| `OLLAMA_CLOUD_API_KEY` | — | Ollama Cloud API key |

See `.env.example` for the full list. The OpenAI router loads `.env` before
initializing settings, so direct uvicorn starts and launcher-managed starts use
the same debug and provider variables.

### Config Files

- `config/models.yaml` — model registry and worker bindings
  Models can list multiple workers under `worker_bindings`. The routing engine
  selects the best available worker by GPU tier (5090 > 4070 > P40), queue depth,
  GPU utilization, and model affinity. Dead or overloaded workers are skipped
  automatically — no manual failover needed.
- `config/cluster.yaml` — cluster topology and node addressing
  `node_hosts` maps each `node_id` to its LAN IP address. This must match the
  IP the worker process uses when registering with the control plane (visible
  via `curl /cluster/workers`). If the IP in `node_hosts` differs from the
  worker's registered `base_url`, the routing engine's first-pass worker lookup
  will fail and fall back to a direct probe — which may also fail if the probe
  times out. Every node in `worker_bindings` should have an entry here:
  ```yaml
  node_hosts:
    node-01: 192.168.1.200     # IP the worker agent binds to
    node-p40-01: 192.168.1.123
  ```
  `local_workers` defines which Ollama ports run on this machine, their GPU
  assignment, and intended workload role. Each port runs an independent
  `ollama serve` process pinned to a specific GPU:
  ```yaml
  local_workers:
    - port: 11434
      gpu_id: 0
      role: coding        # Coding / chat — primary GPU
    - port: 11435
      gpu_id: 1
      role: embeddings    # Embeddings, vision, lightweight models
    - port: 11436
      gpu_id: 0
      role: reasoning     # Long-context / chain-of-thought models
    - port: 11437
      gpu_id: 1
      role: agents        # Agent / tool-calling tasks
  ```
  Start them with:
  ```bash
  # GPU 0 (ports 11434, 11436)
  CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 ollama serve &
  CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11436 ollama serve &

  # GPU 1 (ports 11435, 11437)
  CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve &
  CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11437 ollama serve &
  ```
- `config/routing_rules.yaml` — model routing rules, gateway-safe `model_aliases`, fallback settings
- `config/routing_state.yaml` — runtime routing state (auto-generated, gitignored)
- `config/routing_audit.jsonl` — routing audit log (auto-generated, gitignored)

### Model Aliases

```yaml
model_aliases:
  alias_prefix: AIIH
  entries:
    glm-4.7-flash-q4: glm-4.7-flash:q4_K_M
    gemma4-e4b: gemma4:e4b
```

Clients request `anthropic/AIIH/glm-4.7-flash-q4`; AIIH resolves to the real model.

## Fixed Ports

| Service | Port |
|---------|------|
| Router API (OpenAI) | `8001` |
| Router API (Anthropic) | `8002` |
| Dashboard | `9001` |
| Prometheus Exporter | `9100` |
| Control Plane | `9200` |
| Worker RPC | `9300` |
| Node Agent | `9400` |
| Ollama Worker GPU0 | `11434` |
| Ollama Worker GPU1 | `11435` |

## Directory Layout

```
AetherMesh/
  runtime/              AI Runtime Kernel (v5.0.0)
    kernel.py           AetherKernel bootstrapper
    event_bridge.py     Event bus bridge (graph ↔ runtime)
    launcher/           Service launcher (start/stop/status all services)
    context/            Unified ExecutionContext (12 sub-contexts)
    events/             Typed event bus (26 event types, pub/sub, history)
    state/              Deterministic state machines (5 domains)
    replay/             Execution recording + replay engine
    abi/                7 stable plugin interfaces + lifecycle manager
    intelligence/       Provider capability scoring + execution selector
    memory/             Short-term, semantic, episodic memory
    orchestration/      DAG execution graphs, executor, planner, retry
    multi_agent/        Coordinator, planner agent, worker agent
    observability/      Event bus, tracing, metrics
    gpu_os/             GPU device manager, model scheduler
    security/           Rate limiter, input validator, API key auth, SQLAlchemy DB (User/ApiKey/Session models), JWT, scrypt passwords, user management API
    tools/              Tool runtime + builtin tools
    agents/             Agent loop + lifecycle adapter
    mcp/                MCP gateway
    sessions/           Session management + lifecycle adapter
    responses/          Responses API runtime (full OpenAI Responses format)
    gpu/                GPU scheduling + lifecycle adapter
  router/               Protocol adapters
    openai_router.py    OpenAI-compatible API
    anthropic_router.py Anthropic-compatible API
    responses_router.py Responses API router
    streaming_router.py Streaming adapter
    routing_engine.py   Capability-based routing
    rate_limiter.py     Token bucket rate limiter
  providers/            LLM provider adapters
  control_plane/        Cluster management
  dashboard/            Web dashboard
  metrics/              Observability exporters
  node/                 Node/worker agents
  ai_queue/             Async task queue
  config/               Configuration
  docs/                 Documentation
```

## Sync vs Async Request Flow

### Sync (default)
`POST /v1/chat/completions` without async flags → control plane dispatches → worker serves → release + telemetry.

### Async queue mode
Set `"async": true`, `"background": true`, or `"queue": true` in request body.
Router enqueues via `/cluster/tasks` → returns `task_id` + `poll_url`.
Task worker (`python -m ai_queue.task_worker`) executes and updates status.

## Routing & Scheduling

### Provider Selection
1. Gateway-safe model aliases resolved first
2. Model provider resolved from `config/models.yaml`
3. Fallback by model name prefix: `*cloud` → Ollama Cloud, `gemini*` → Gemini, `gpt*/o1*/o3*/o4*` → OpenAI, `meta/`/`mistralai/`/`nvidia/` etc. → NVIDIA NIM, else → Ollama

### Worker Assignment
Scheduler picks from healthy workers by: lowest GPU utilization → lowest queue size → random tie-break.
Workers excluded at GPU utilization ≥ 85% or queue exceeding `AIIH_MAX_WORKER_QUEUE`.
Sync dispatches carry an `assignment_id`; routers release the same assignment on completion.
If a router exits mid-request, the control plane reclaims unreleased assignments after
`AIIH_WORKER_ASSIGNMENT_TTL` seconds so `queue_size` does not remain stuck and cause
false `worker_queue_full` 429s.
GPU saturation (utilization ≥ 85%) is reported as worker unavailability instead of queue capacity.

### Hierarchical Fallback
- Tier 1 (S-Tier): high-performance GPU (RTX 5090)
- Tier 2 (A-Tier): mid-performance GPU (RTX 4070 Ti)
- Tier 3 (B-Tier): high-capacity GPU (Tesla P40)

Saturated tiers (GPU ≥ 85%) divert to the next tier.
OpenAI-compatible sync routing includes local fallback chains for heavy 5090-bound models
such as `qwen3.6:27b` and `qwen3.6:35b`, allowing requests to fall back to smaller
models on other available local workers when the primary GPU is busy.

## Monitoring & Dashboard

Dashboard at `http://localhost:9001`:
- Ops overview: alerts, health signals
- Provider health, request metrics, local-only mode, model overrides, routing audit log
- **Model Alias Map**: gateway model → AIIH alias → target model → provider → worker
- **Cloud Providers**: connection status, model count, latency for NVIDIA NIM, Ollama Cloud, OpenAI, Gemini
- **Charts**: latency trends, request rates, GPU utilization, model usage, worker load, tokens/sec, TTFT, GPU memory

All dashboard API endpoints (`/api/*`) are consolidated under a single `APIRouter(prefix="/api")`
in `dashboard/dashboard_server.py`, with the dashboard static JS extracted to
`dashboard/static/dashboard.js` and CSS inlined into the HTML template.

Prometheus exporter at `http://localhost:9100/metrics`.

### Metrics

Control plane metrics snapshot:
`request_total`, `request_latency_ms_avg/p95/p99`, `endpoint_latency`, `worker_usage`,
`queue_length`, `model_usage`, `provider_status`, `error_count`.

Common error codes: `provider_timeout`, `runner_stopped`, `provider_unreachable`,
`worker_unavailable`, `control_plane_unavailable`, `queued_sync_not_supported`.

## Validation

```bash
curl http://127.0.0.1:9200/cluster/nodes
curl http://127.0.0.1:9200/cluster/workers
curl http://127.0.0.1:8001/v1/models

# Chat
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{"model":"qwen3.5:27b","messages":[{"role":"user","content":"hello"}]}'

# Responses API
curl http://127.0.0.1:8001/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{"model":"qwen3.5:27b","input":"hello","instructions":"Be concise"}'

# Embeddings
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{"model":"nomic-embed-text:latest","input":"test"}'

# Async
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{"model":"qwen3.5:27b","async":true,"messages":[{"role":"user","content":"run async"}]}'
```

---

See `docs/` for architecture overview, evolution plan, and detailed API documentation.
