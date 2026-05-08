# AetherMesh Runtime Platform — 重構計畫 (v1.0)

> 根據您提供的重構需求與專案現有程式碼分析後擬定。
> 請確認後告知，我將依序執行。

---

## 當前專案現狀

| 指標 | 數值 |
|---|---|
| Python 原始檔 | 55 個 |
| 總行數 (自有程式碼) | ~11,000 SLOC |
| 最大檔案 | `index.html` (2,037)、`anthropic_router.py` (1,331)、`openai_router.py` (953)、`ollama_adapter.py` (830) |
| 測試檔 | 6 個 (`test_tool_call_normalizer.py` 等) |

### 尚未存在的目錄
- `runtime/` ❌ — 核心執行期
- `tools/` ❌ — tool 邏輯散落在 router/ 與 providers/ 中
- `mcp/` ❌
- `agents/` ❌
- `sessions/` ❌
- `security/` ❌
- `protocols/` ❌

### 主要架構問題

1. **router/ 過度膨脹**: `anthropic_router.py` + `openai_router.py` 合計 ~2,284 行，混合 protocol 轉換、工具解析、路由決策、metrics
2. **Tool 邏輯重複**: 工具解析程式碼在 `ollama_adapter.py` 和 `anthropic_router.py` 中重複實作
3. **Web search 內嵌在 router**: `router/web_server_tools.py` 是 built-in tool 但放在錯誤位置
4. **無正式 Tool Runtime**: 模型直接產生工具執行要求，平台不負責執行
5. **無 MCP Gateway**: MCP 客戶端需直連 MCP server
6. **無 Agent Runtime**: 無多步驟執行、planner/executor 分離
7. **無 Session Runtime**: 無持久化 session、resumable execution
8. **無 Security Layer**: 無工具沙箱、prompt firewall
9. **Provider 能力非正式**: capability 是 hardcoded score matrix

---

## Phase 1 — Platform Repositioning ✅ (已完成)

- [x] README 標題/副標題更新
- [x] Dashboard branding 更新
- [x] API title 更新 (6 個 FastAPI app)
- [x] .env.example、systemd、scripts 中的名稱更新
- [x] AIIH 保留為 `Aether Intelligent Infrastructure Hub`

---

## Phase 2 — Runtime-Centric Architecture

**目標**: 建立 `runtime/` 為核心的目錄結構，將 orchestration 邏輯從 `router/` 移出。

### 新增目錄結構

```
runtime/
  __init__.py
  tools/
    __init__.py
    tool_registry.py       # 工具登記
    tool_runtime.py        # 工具執行生命週期
    tool_executor.py       # 工具執行器
    tool_result.py         # 工具結果模型
    builtin/
      __init__.py
      web_search.py        # 從 router/web_server_tools.py 遷移
      web_fetch.py         # 從 router/web_server_tools.py 遷移
      shell.py
      filesystem.py
      python.py
      http_request.py
    web_search/
      __init__.py
      search_provider.py   # SearchProvider interface
      tavily.py            # Tavily Search API
      serper.py            # Serper.dev Search API
      duckduckgo.py        # Fallback (DuckDuckGo)
  agents/
    __init__.py
    agent_context.py
    agent_loop.py
    agent_step.py
    agent_result.py
  mcp/
    __init__.py
    mcp_registry.py
    mcp_session_manager.py
    mcp_capability.py
    mcp_auth.py
    mcp_sandbox.py
    mcp_tool_bridge.py
  sessions/
    __init__.py
    session_store.py
    session_manager.py
  responses/
    __init__.py
    response_runtime.py
  gpu/
    __init__.py
    vram_scheduler.py
    model_affinity.py
    warm_pool.py
  security/
    __init__.py
    tool_sandbox.py
    prompt_firewall.py
    secret_detection.py
    audit_log.py
  orchestration/
    __init__.py
    execution_lifecycle.py
```

### 搬遷計畫

| 目前位置 | 搬遷到 | 說明 |
|---|---|---|
| `router/routing_engine.py` | `runtime/orchestration/routing_engine.py` | 路由決策是 runtime 行為 |
| `router/capabilities.py` | `runtime/orchestration/capabilities.py` | 能力推導是 runtime 行為 |
| `router/web_server_tools.py` | `runtime/tools/builtin/web_search.py` + `web_fetch.py` | Web tool 是 runtime builtin |
| `router/server_tool_policy.py` | `runtime/security/tool_policy.py` | 工具政策是安全層 |
| `router/tool_call_normalizer.py` | `runtime/tools/tool_normalizer.py` | 工具解析是 tool runtime 的一部分 |
| `router/content_blocks.py` | `runtime/tools/content_blocks.py` | content block 轉譯是 tool runtime |
| `router/anthropic_sse_builder.py` | `protocols/anthropic/sse_builder.py` | SSE 格式是 protocol 層 |

### router/ 精簡後

```
router/
  __init__.py
  openai_router.py         # 僅 protocol 轉換 + 轉發到 runtime
  anthropic_router.py      # 僅 protocol 轉換 + 轉發到 runtime
  responses_router.py      # 僅 Responses API protocol
  rate_limiter.py          # 保留 (middleware)
  streaming_router.py      # 保留 (SSE helper)
```

---

## Phase 3 — Tool Runtime

**現狀**: tool call 解析散落在 `ollama_adapter.py` 與 `anthropic_router.py`；沒有統一 Tool Runtime。

### 新架構

```
Client Request
  → Protocol Adapter (router/) 識別 tool calls
  → runtime/tools/tool_runtime.py  統一管理
    → tool_registry.py 查詢可用工具
    → tool_executor.py 執行工具 (含 sandbox)
    → tool_result.py 包裝結果
  → 結果注入回 model context
  → 繼續 generation
```

### 核心模型 (`runtime/tools/tool_result.py`)

```python
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
    source_provider: str
    source_model: str

@dataclass
class ToolResult:
    call: ToolCall
    output: Any
    is_error: bool
    duration_ms: float
```

### Built-in 工具 (`runtime/tools/builtin/`)

- `web_search.py` — 從 `router/web_server_tools.py` 遷移
- `web_fetch.py` — 從 `router/web_server_tools.py` 遷移
- `shell.py` — 安全 shell 執行 (受 sandbox 控制)
- `filesystem.py` — 安全的檔案讀寫
- `python.py` — Python 程式碼執行 (sandboxed)
- `http_request.py` — HTTP 請求工具

### 工具政策 (`runtime/security/tool_policy.py`)

從 `router/server_tool_policy.py` 遷移並擴充，支援 permissions-based policy。

---

## Phase 4 — Web Search Runtime

**現狀**: `router/web_server_tools.py` 使用 DuckDuckGo HTML scraper，搜尋能力有限。

### 支援的搜尋服務商

| Provider | 類型 | API Key 需求 |
|---|---|---|
| **Tavily** | 雲端 API | `TAVILY_API_KEY` |
| **Serper.dev** | 雲端 API | `SERPER_API_KEY` |
| DuckDuckGo | Fallback (無需 key) | 無 |

### 新架構

```
runtime/tools/web_search/
  __init__.py
  search_provider.py   # SearchProvider ABC
  tavily.py            # Tavily Search API
  serper.py            # Serper.dev Search API
  duckduckgo.py        # Fallback (DuckDuckGo)
```

執行流程：
```
Client → AetherMesh Tool Runtime → SearchProvider → 工具結果注入 model context → 繼續 generation
```

---

## Phase 5 — MCP Gateway

**現狀**: 無 MCP 支援。

AetherMesh 成為 MCP Gateway。客戶端（Claude Code 等）連接到 AetherMesh，由 AetherMesh 代理到後端 MCP Servers。

```
Claude Code
  ↓ (MCP protocol)
AetherMesh MCP Gateway (runtime/mcp/)
  ├── filesystem MCP server
  ├── git MCP server
  ├── browser MCP server
  ├── shell MCP server
  └── ...
```

### 元件

| 檔案 | 職責 |
|---|---|
| `mcp_registry.py` | 註冊/發現 MCP servers |
| `mcp_session_manager.py` | 管理 client→MCP session 生命週期 |
| `mcp_capability.py` | MCP capability negotiation |
| `mcp_auth.py` | MCP 認證中介層 |
| `mcp_sandbox.py` | MCP tool 沙箱 |
| `mcp_tool_bridge.py` | MCP tool ↔ Tool Runtime bridge |

---

## Phase 6 — Agent Runtime

**現狀**: 無 agent 支援。

```
runtime/agents/
  agent_context.py   # AgentExecutionContext — 包含 session、memory、tools
  agent_loop.py      # AgentLoop — 多步驟執行迴圈
  agent_step.py      # AgentStep — 單一步驟 (think → act → observe)
  agent_result.py    # AgentResult — 執行結果
```

### Agent 執行流程

```
AgentLoop.run(context):
  while not done:
    step = plan_next_action(context)
    result = execute_step(step)
    context.add_step(result)
  return context.final_result()
```

支援 pattern:
- **Single-agent**: 單一 agent 逐步執行
- **Multi-agent**: 多個 agent 協作
- **Planner-worker**: planner 分配任務、worker 執行

---

## Phase 7 — Responses API Native

**現狀**: `router/openai_router.py` 已有 `/v1/responses` endpoint，但實質只是轉發到 provider adapter。

### 目標

```
runtime/responses/
  response_runtime.py   # 核心 Responses 執行引擎
```

支援：
- **Parallel tool calls**: 一次回應含多個 tool calls
- **Reasoning traces**: 推理過程可串流
- **Background execution**: 非同步執行背景任務
- **Resumable execution**: 中斷後可恢復
- **Structured outputs**: JSON schema 輸出
- **Streaming tool deltas**: 工具呼叫的 delta 事件

---

## Phase 8 — Provider Capability Registry

**現狀**: `router/capabilities.py` 用 hardcoded `CAPABILITY_PROVIDER_SCORES` matrix。

### 目標

改為 provider 自述能力註冊：

```python
# providers/registry.py (新增)
class ProviderCapabilityRegistry:
    def register(self, name: str, capabilities: set[Capability])
    def get_providers_for(self, required: set[Capability]) -> list[ProviderEntry]
    def score_provider(self, name: str, required: set[Capability]) -> float
```

Routing engine 改為根據以下維度評分：
1. **Capabilities** (必要條件 match)
2. **Latency** (即時測量)
3. **Health** (heartbeat + probe + circuit breaker)
4. **GPU pressure** (VRAM 使用率 + queue depth)
5. **Cost** (雲端 vs 本機)
6. **Tool requirements** (是否需要特定工具支援)

---

## Phase 9 — GPU Runtime

**現狀**: GPU 管理散落在 `cluster/gpu_discovery.py`、`control_plane/scheduler.py`、`cluster/load_balancer.py`。

### 目標

```
runtime/gpu/
  vram_scheduler.py    # VRAM-aware 調度
  model_affinity.py    # Model 親和性 (已載入模型優先)
  warm_pool.py         # 模型熱池
```

### 調度維度

| 維度 | 說明 |
|---|---|
| VRAM | 依剩餘 VRAM 分配 |
| Model locality | 已載入 model 的 GPU 優先 |
| KV cache | cache 重複利用 |
| Queue depth | worker 佇列深度 |
| Latency | 歷史延遲 |
| GPU tier | 5090 > 4070Ti > P40 > M4 |

硬體拓撲：
- RTX 5090 (32GB) — 高效能主力
- RTX 4070 Ti SUPER (16GB) — 中階推理
- Tesla P40 (24GB) × N — 高容量批次
- Apple Silicon M4 (64GB) — 大記憶體模型

---

## Phase 10 — Session Runtime

**現狀**: 無 session 概念，每次請求獨立。

```
runtime/sessions/
  session_store.py     # Session 儲存 (支援 Redis/memory)
  session_manager.py   # Session 生命週期管理
```

支援：
- **Persistent sessions**: 跨請求保留
- **Resumable sessions**: 中斷後從上次狀態恢復
- **Multi-client sessions**: 多客戶端共享 session
- **Agent memory**: Agent 可讀寫的持久化記憶
- **Conversation state**: 完整對話狀態管理

---

## Phase 11 — Security Layer

**現狀**: 無正式安全層。

```
runtime/security/
  tool_sandbox.py        # 工具執行沙箱 (subprocess/isolation)
  prompt_firewall.py     # Prompt 注入檢測
  secret_detection.py    # 輸出中的 secret 偵測
  tool_policy.py         # 工具執行政策 (從 server_tool_policy.py 遷移)
  audit_log.py           # 工具執行審計
```

安全層涵蓋：
- **Tool sandbox**: 限制 shell/filesystem/python 工具的作用範圍
- **Prompt firewall**: 檢測 prompt injection 試圖
- **Secret detection**: 防止模型輸出 API key/密碼
- **MCP permission controls**: MCP tool 存取控制
- **Filesystem isolation**: 限制可讀寫的目錄
- **Timeout controls**: 工具執行 timeout
- **Execution audit logs**: 所有工具執行記錄

---

## Phase 12 — Observability

**現狀**: 兩套 metrics：`MetricsStore` (cluster) + `RequestMetricsCollector` (per-request)。

### 新增 metrics 維度

| 類別 | 新增指標 |
|---|---|
| Tool execution | tool call count, duration, error rate, tool name distribution |
| Agent execution | step count, agent type, completion rate, retry count |
| MCP | session count, tool calls, error rate, latency |
| Reasoning | thinking tokens, reasoning steps, budget usage |
| Session | active sessions, session duration, message count |
| Provider capability | capability match rate, fallback rate, cooldown events |
| GPU | VRAM fragmentation, model load/unload rate, KV cache hit rate |

Dashboard 演進為 **AetherMesh Control Center**，新增 Tools / Agents / MCP / Sessions / Security 面板。

---

## Phase 13 — Router Simplification

**原則**: router/ 只負責 protocol 轉換，所有商業邏輯歸 runtime/。

### 重構後 router/

```
router/
  __init__.py
  openai/
    __init__.py
    chat_adapter.py      # /v1/chat/completions protocol 轉換
    responses_adapter.py  # /v1/responses protocol 轉換
    models_adapter.py     # /v1/models
    embeddings_adapter.py # /v1/embeddings
    rerank_adapter.py     # /v1/rerank
  anthropic/
    __init__.py
    messages_adapter.py   # /v1/messages protocol 轉換
    sse_builder.py        # 從 router/anthropic_sse_builder.py 遷移
  mcp/
    __init__.py
    mcp_adapter.py        # MCP protocol endpoint
  rate_limiter.py         # 保留
  streaming_router.py     # 保留
```

每個 adapter 只做：
1. **解析** 請求格式
2. **轉換** 為內部 runtime 格式
3. **呼叫** runtime 對應服務
4. **轉換** 回響應格式

---

## Phase 14 — Clean Architecture

### 最終目標目錄結構

```
cli/                          # CLI client (新增)
  __init__.py
  aethermesh_cli.py
clients/                      # Client SDK (新增)
  __init__.py
  openai_sdk.py
  anthropic_sdk.py
  mcp_sdk.py
runtime/                      # 核心執行引擎 (新增)
  tools/
  agents/
  mcp/
  sessions/
  responses/
  gpu/
  security/
  orchestration/
protocols/                    # Protocol adapters (從 router/ 重構)
  openai/
  anthropic/
  mcp/
providers/                    # Provider adapters (保留)
  base.py
  ollama_adapter.py
  openai_adapter.py
  gemini_adapter.py
  nvidia_nim_adapter.py
  ollama_cloud_adapter.py
  registry.py                 # Provider capability registry (新增)
  http_client.py
control_plane/                # Cluster management (保留，精簡)
  cluster_manager.py
  worker_registry.py
  node_registry.py
  scheduler.py
dashboard/                    # Dashboard (保留，擴充)
  dashboard_server.py
  templates/
  static/
metrics/                      # Metrics (保留，擴充)
  metrics.py
  request_metrics.py
  prometheus_exporter.py
cluster/                      # Cluster services (保留)
  circuit_breaker.py
  load_balancer.py
  gpu_discovery.py
  health_checker.py
node/                         # Node services (保留)
  worker_agent.py
  node_agent.py
ai_queue/                     # Async queue (保留)
  redis_queue.py
  task_worker.py
config/                       # Configuration (保留，精簡)
  settings.py
  structured_logging.py
  models.yaml
  cluster.yaml
  routing_rules.yaml
tests/                        # Tests (擴充)
scripts/                      # Scripts (保留)
docs/                         # 新增文件目錄
  architecture/
  runtime/
  mcp/
  tools/
  providers/
  gpu/
  security/
```

### 避免的反模式

- ❌ **Giant router files**: 任何檔案超過 500 行應拆分
- ❌ **Provider-specific hacks**: Provider 差異應封裝在 adapter 內
- ❌ **Duplicated tool parsing**: 統一由 `runtime/tools/tool_normalizer.py` 處理
- ❌ **Provider-specific execution loops**: 執行迴圈統一由 `runtime/` 管理

---

## Phase 15 — Documentation

### 新增文件

```
docs/
  architecture/
    overview.md
    runtime-lifecycle.md
  runtime/
    tool-lifecycle.md
    agent-lifecycle.md
    session-lifecycle.md
  mcp/
    gateway-architecture.md
    bridge-pattern.md
  tools/
    builtin-tools.md
    tool-policy.md
    web-search.md
  providers/
    capability-registry.md
    adding-new-provider.md
  gpu/
    scheduling.md
    topology.md
  security/
    sandbox.md
    policies.md
```

---

## 執行順序與相依性

```
Phase 1: Platform Repositioning  ✅ 已完成

Phase 2: Runtime-Centric Architecture
  └─ 相依: 無 (建立空目錄結構，可先做)
  └─ 風險: 低 (新增檔案，不影響現有功能)

Phase 3: Tool Runtime
  └─ 相依: Phase 2
  └─ 風險: 中 (需確保向後相容 tool call 解析)

Phase 4: Web Search Runtime
  └─ 相依: Phase 3 (需 Tool Runtime 執行)
  └─ 風險: 低 (替換現有 DuckDuckGo 實作)

Phase 5: MCP Gateway
  └─ 相依: Phase 3 (需 Tool Runtime bridge)
  └─ 風險: 中 (新 protocol，需測試)

Phase 6: Agent Runtime
  └─ 相依: Phase 3, Phase 10 (session)
  └─ 風險: 中 (新概念，需設計 iteration)

Phase 7: Responses API Native
  └─ 相依: Phase 3, Phase 10
  └─ 風險: 高 (涉及執行模型改變)

Phase 8: Provider Capability Registry
  └─ 相依: Phase 2
  └─ 風險: 中 (需修改 routing engine)

Phase 9: GPU Runtime
  └─ 相依: Phase 2
  └─ 風險: 低 (從 cluster/ 重構搬遷)

Phase 10: Session Runtime
  └─ 相依: Phase 2
  └─ 風險: 中 (新狀態管理)

Phase 11: Security Layer
  └─ 相依: Phase 3 (需保護 tool execution)
  └─ 風險: 中 (安全需謹慎設計)

Phase 12: Observability
  └─ 相依: Phase 3~11 (需所有 runtime 上線後擴充)
  └─ 風險: 低 (擴充 metrics)

Phase 13: Router Simplification
  └─ 相依: Phase 2~11 (所有邏輯移出後才可精簡)
  └─ 風險: 高 (需確保 protocol 完全相容)

Phase 14: Clean Architecture
  └─ 相依: Phase 13 (最終目錄整理)
  └─ 風險: 低 (目錄搬遷)

Phase 15: Documentation
  └─ 相依: Phase 2~14
  └─ 風險: 低
```

---

## 風險評估

| 風險 | 等級 | 緩解 |
|---|---|---|
| 重構中斷現有功能 | 高 | 每個 Phase 皆有完整測試；逐步 rollout |
| Router 精簡破壞 Claude Code 相容性 | 高 | Protocol adapter 需 100% 通過相容測試 |
| MCP Gateway 增加延遲 | 中 | 非同步 non-blocking 架構 |
| Agent Runtime 設計過度工程 | 中 | 先支援 single-agent，再擴充 multi-agent |
| Tool Runtime 改變 tool call 行為 | 中 | 確保 backward-compatible 的 tool normalizer |
| Session 持久化增加複雜度 | 中 | Redis 為預設 backend，memory fallback |

---

*此計劃為 v1.0 草案。確認後將依 Phase 順序逐步執行。*
