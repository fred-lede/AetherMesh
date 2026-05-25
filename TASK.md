# AetherMesh Runtime Platform — 重構進度

> 基於 [REFACTOR_PLAN.md](REFACTOR_PLAN.md) 執行，記錄各 Phase 完成狀態。

---

## Phase 1 — Platform Repositioning ✅ (2026-05-07)
- [x] README 標題/副標題更新
- [x] Dashboard branding 更新
- [x] API title 更新 (6 個 FastAPI app)
- [x] .env.example、systemd、scripts 中的名稱更新
- [x] AIIH 保留為 Aether Intelligent Infrastructure Hub

## Phase 2/3 — Runtime-Centric Architecture + Tool Runtime 🟡 (2026-05-07)
- [x] runtime/ 目錄結構建立 (tools, agents, mcp, sessions, responses, gpu, security, orchestration)
- [x] router/ → runtime/ 全部 migration re-export
- [x] Web Search Runtime (Tavily + Serper + DuckDuckGo)
- [x] Tool Runtime Core (registry, executor, runtime, result, normalizer)
- [x] Tool Policy 遷移
- [x] **4 個 builtin tools** (shell, filesystem, python, http_request)
- [x] builtin/__init__.py auto-register
- [x] Dashboard crash fix 驗證 (Windows Jinja2 ✅)

## Phase 5 — MCP Gateway 🟡
- [x] mcp_registry.py
- [x] mcp_session_manager.py
- [x] mcp_tool_bridge.py
- [x] mcp_capability.py
- [x] mcp_auth.py
- [x] mcp_sandbox.py

## Phase 6 — Agent Runtime 🟡
- [x] agent_loop.py (refactored to use separated classes)
- [x] agent_context.py
- [x] agent_step.py
- [x] agent_result.py

## Phase 7 — Responses API Native ✅
- [x] response_runtime.py

## Phase 8 — Provider Capability Registry ✅
- [x] providers/registry.py (Capability enum + entry dataclass)
- [x] ProviderCapabilityRegistry class (register/get_providers_for/score_provider)
- [x] Extended scoring: GPU pressure, cost, tool requirements

## Phase 9 — GPU Runtime ✅
- [x] vram_scheduler.py, model_affinity.py, warm_pool.py

## Phase 10 — Session Runtime ✅
- [x] session_store.py, session_manager.py

## Phase 11 — Security Layer ✅
- [x] tool_sandbox.py, prompt_firewall.py, secret_detection.py, tool_policy.py, audit_log.py

## Phase 12 — Observability 🟡
- [x] Tool execution metrics
- [x] Agent metrics
- [x] MCP metrics
- [x] Provider capability metrics
- [x] Reasoning metrics (thinking tokens, reasoning steps, budget)
- [x] GPU metrics (VRAM fragmentation, load/unload rate, KV cache hit)
- [x] Session metrics (duration, message count)

## Phase 13 — Router Simplification ✅
- [x] router/openai/ (chat_adapter, responses_adapter, models_adapter, embeddings_adapter, rerank_adapter)
- [x] router/anthropic/ (messages_adapter)
- [x] router/mcp/ (mcp_adapter)
- [x] Original flat files re-export from new locations (backward compat)

## Phase 14 — Clean Architecture ✅
- [x] cli/ (aethermesh_cli.py)
- [x] clients/ (openai_sdk, anthropic_sdk, mcp_sdk)
- [x] protocols/ 補完 (openai/chat.py, mcp/protocol.py)

## Phase 15 — Documentation ✅
- [x] docs/architecture/overview.md
- [x] docs/architecture/runtime-lifecycle.md
- [x] docs/runtime/tool-lifecycle.md
- [x] docs/runtime/agent-lifecycle.md
- [x] docs/runtime/session-lifecycle.md
- [x] docs/mcp/gateway-architecture.md
- [x] docs/mcp/bridge-pattern.md
- [x] docs/tools/builtin-tools.md
- [x] docs/tools/tool-policy.md
- [x] docs/tools/web-search.md
- [x] docs/providers/capability-registry.md
- [x] docs/providers/adding-new-provider.md
- [x] docs/gpu/scheduling.md
- [x] docs/gpu/topology.md
- [x] docs/security/sandbox.md
- [x] docs/security/policies.md

## Phase 7 — Kernel Stabilization (Execution Context) ✅
- [x] `runtime/context/` — Unified `ExecutionContext` with 12 sub-context states (provider, tool, gpu, session, stream, memory, security, graph, trace, etc.)
- [x] `runtime/events/` — Typed event bus with 26 event types, pub/sub, history, async + sync dispatch
- [x] `runtime/state/` — Deterministic state machines with validated transitions across 5 domains (execution, stream, session, agent, provider)
- [x] `runtime/event_bridge.py` — Bidirectional bridge between graph event bus and runtime event bus

## Phase 8 — Execution Replay + Runtime ABI ✅
- [x] `runtime/replay/` — `ExecutionRecorder` (event stream + snapshots + serialization), `ReplayEngine` (load/replay/export), `TraceRebuilder`
- [x] `runtime/abi/` — 7 stable `RuntimeComponent` plugin interfaces (agent, gpu, memory, provider, stream, tool, runtime_contract)
- [x] `runtime/abi/lifecycle_manager.py` — `RuntimeLifecycleManager` with `initialize/start/pause/resume/cancel/shutdown` for all components
- [x] `runtime/kernel.py` — `AetherKernel` bootstrapper with `create_execution/start/pause/resume/cancel/fail/complete/shutdown`

## Phase 8b — Full Responses API Support ✅
- [x] `runtime/responses/` — `ResponseRuntime`, response models (`ResponseObject`, `ResponseStatus`, `ResponseUsage`, `OutputItem`, `ContentPart`)
- [x] `runtime/responses/input_converter.py` — Convert chat/Anthropic formats to Responses format
- [x] `runtime/responses/output_converter.py` — Convert Responses format to chat/streaming format
- [x] `runtime/responses/response_stream.py` — SSE streaming for Responses API responses
- [x] `router/responses_router.py` — FastAPI router with CRUD endpoints (`POST/GET/DELETE/PATCH /v1/responses`)
- [x] All 5 provider adapters (ollama, openai, gemini, nvidia_nim, ollama_cloud) implement `responses` capability
- [x] Auto-conversion: native passthrough for OpenAI, format conversion for all other providers

## Phase 16 — README Full Rewrite ✅ (2026-05-08)
- [x] Rewrote 983-line v4 README to 382-line v5 kernel-focused README
- [x] Architecture layer diagram (kernel-centric instead of old cluster topology)
- [x] Kernel Core documentation (AetherKernel, ExecutionContext, EventBus, StateMachine, ReplayEngine, LifecycleManager)
- [x] Python quick-start for standalone kernel usage
- [x] Responses API documentation + validation curl command
- [x] Condensed deployment, config, routing, monitoring sections
- [x] Removed v4-era maintenance log, baseline benchmarks, systemd/launchd edge cases

## Phase 17 — Dashboard Template Modernization ✅ (2026-05-08)
- [x] Consolidated all `/api/*` routes into `APIRouter(prefix="/api")` in `dashboard/dashboard_server.py`
- [x] Extracted inline JavaScript from `templates/index.html` to `static/dashboard.js`
- [x] Inlined CSS from `static/style.css` directly into `templates/index.html`
- [x] Reduced `index.html` from ~1356 lines of JS/CSS/HTML to 820 lines (pure template)
- [x] Added `dashboard.js` as a static file served via `app.get("/static/{file_path:path}")`
- [x] No functional changes — all 18 `/api/*` endpoints maintain identical paths

## Phase 18 — Service Launcher ✅ (2026-05-08)
- [x] Created `runtime/launcher/` package with `python -m runtime.launcher` entry point
- [x] Unified launcher starts all 8 services in a single process group from one terminal
- [x] Each service writes output to `logs/<name>.log` for independent debugging
- [x] Supports `start`, `stop`, `status`, `restart` commands with per-service targeting
- [x] Loads `.env` automatically before launching services
- [x] Graceful Ctrl+C shutdown — SIGTERM all subprocesses, SIGKILL after 5s timeout
- [x] Updated README.md with launcher usage, boot startup (systemd/launchd/Task Scheduler)
- [x] Updated old systemd files from `ai_inference_hub` → `__ROOT_DIR__` placeholders
- [x] Created `systemd/aiih-launcher.service` and `launchd/com.aiih.launcher.plist.example`
- [x] README: expanded Installation (venv, Python version, system deps), added Profiles section
- [x] README: added Remote Worker Nodes section (multi-node cluster setup guide)
- [x] Boot startup: systemd service (Ubuntu), launchd plist (macOS), Task Scheduler (Windows)
- [x] Updated old systemd files from `ai_inference_hub` to `__ROOT_DIR__` placeholders

## Phase 19 — Database-Backed Auth Subsystem ✅ (2026-05-10)
- [x] SQLAlchemy + SQLite setup (`runtime/security/database.py`)
- [x] User / ApiKey / Session ORM models (`runtime/security/models.py`)
- [x] Password hashing with scrypt (`runtime/security/auth/password.py`)
- [x] JWT token create/decode (`runtime/security/auth/jwt.py`)
- [x] Admin bootstrap from env vars (`runtime/security/auth/admin_bootstrap.py`)
- [x] API key generation / validation / CRUD (`runtime/security/auth/api_key.py`)
- [x] FastAPI auth dependencies (`runtime/security/auth/dependencies.py`)
- [x] Middleware updated: env var + DB fallback for API key verification
- [x] Dashboard API Key management routes + UI (list / create / revoke)
- [x] Launcher calls bootstrap_admin on startup
- [x] `.env.example` — added `AIIH_ADMIN_EMAIL`, `AIIH_ADMIN_PASSWORD`, `AIIH_DB_PATH`
- [x] `POST /api/auth/login` — JWT login endpoint (email + password)
- [x] User CRUD routes: `GET /api/users`, `POST /api/users`, `PATCH /api/users/{id}`, `DELETE /api/users/{id}`
- [x] Dashboard Users section — list / create / edit (name, role, password) / delete

## Phase 20 — Worker Assignment Resilience ✅ (2026-05-19)
- [x] Fixed execution selector rerank so it preserves the routing engine's concrete `provider/model/worker` decision instead of producing `ollama` with no worker.
- [x] Added sync worker assignment IDs and release tracking so router crashes cannot leave `queue_size` inflated indefinitely.
- [x] Added `AIIH_WORKER_ASSIGNMENT_TTL` to reclaim unreleased assignments after a bounded lease period.
- [x] Distinguished GPU saturation from queue capacity so saturated workers no longer surface as misleading `worker_queue_full` 429s.
- [x] Added local fallback chains for `qwen3.6:27b` and `qwen3.6:35b` to route around a busy 5090 when smaller workers are available.
- [x] Updated launcher `.env` loading to override inherited environment values for deterministic service startup.

## Phase 21 — Embeddings Routing Normalization ✅ (2026-05-24)
- [x] Registered the OpenAI-compatible `POST /v1/embeddings` route in `router/openai_router.py`.
- [x] Classified OpenAI embedding payloads (`input` without `messages`) as the `embeddings` capability instead of `chat`.
- [x] Normalized Ollama and Ollama Cloud embedding response shapes: `embeddings`, legacy `embedding`, and OpenAI-style `data[].embedding`.
- [x] Verified `nomic-embed-text-v2-moe:latest` remains an embeddings-only model path and returns non-empty 768-dimensional vectors through AetherMesh.

## Phase 22 — Responses API Multi-Turn Tool Loop 🟡 (2026-05-24)
### 已完成
- [x] 建立計畫文件 `RESPONSES_TOOL_LOOP_PLAN.md`
- [x] `response_models.py` — 加 `REQUIRES_ACTION` status, `FunctionCallStatus`, `FUNCTION_CALL_OUTPUT` input type, `FUNCTION_CALL` output type
- [x] `make_function_call_output()` / `make_function_call_output_item()` helpers
- [x] `input_converter.py` — 更新 `_parse_input_item` / `_input_item_to_messages` 支援 `function_call_output`
- [x] `tool_loop.py` — `ResponsesToolLoop` 核心實作 (sync + streaming), 最大 16 輪, tool 執行 + 回注 messages
- [x] `openai_handler.py` — `handle_responses()` 加 tool loop 分支 (provider != "openai" 時啟用)
- [x] `openai_handler.py` — `handle_streaming_responses()` 加 streaming tool loop 分支
- [x] `openai_handler.py` — 加 `_resolve_max_turns()` / `_record_response_usage()` helpers
- [x] `__init__.py` — 更新 export 清單
- [x] 全部檔案 `py_compile` 語法驗證通過
- [x] 全部 import 驗證通過

### 待完成
- [ ] 端對端測試驗證 `/v1/responses` with tools → tool execution → completed

---

## 執行記錄

| Date | Phase | Action |
|---|---|---|
| 2026-05-07 | All | 15-phase refactoring 完成 (平台更名 + runtime/ + protocols/) |
| 2026-05-07 | Phase 2/3 | Dashboard Jinja2 crash fix (pre-compiled templates) |
| 2026-05-07 | Phase 2/3 | Windows Dashboard 驗證通過 |
| 2026-05-08 | Phase 7-8 | Kernel stabilization 完成 (context/events/state/replay/abi/kernel.py) |
| 2026-05-08 | Phase 8b | Full OpenAI Responses API 支援完成 (runtime/responses/ + router/responses_router.py + 5 provider adapters) |
| 2026-05-08 | Phase 16 | README.md 完整重寫為 v5 kernel-focused 版本 (983→382 lines) |
| 2026-05-08 | Phase 17 | Dashboard template modernization: JS extracted, CSS inlined, API routes consolidated |
| 2026-05-08 | Phase 18 | Service launcher: `python -m runtime.launcher` starts all 8 services in one terminal + boot startup configs + multi-node docs |
| 2026-05-10 | Phase 19 | Database-backed auth subsystem: SQLAlchemy models + scrypt passwords + JWT + DB API key validation + Dashboard UI for key management |
| 2026-05-10 | Phase 19 | User management CRUD: `POST /api/auth/login`, `GET/POST /api/users`, `PATCH/DELETE /api/users/{id}`, Dashboard Users UI |
| 2026-05-11 | Phase 8 | ProviderCapabilityRegistry + extended scoring (GPU pressure, cost, tool) verified complete |
| 2026-05-19 | Phase 20 | Worker assignment resilience: leased dispatch/release, stuck queue reclamation, rerank worker preservation, GPU saturation classification, qwen3.6 local fallback |
| 2026-05-24 | Phase 21 | Embeddings routing normalization: `/v1/embeddings` route, embeddings capability detection, Ollama response-shape normalization |
| 2026-05-24 | Phase 22 | Responses API Multi-Turn Tool Loop: `ResponsesToolLoop` sync+streaming, tool execution + message reinjection, RouterService integration |
| 2026-05-25 | Phase 23 | Responses client compatibility: `input_text` input normalization, `response.output_text.delta` streaming, completed output assembly |
| 2026-05-25 | Phase 23b | Responses client display compatibility: expose assembled assistant text as top-level `output_text` |
| 2026-05-25 | Phase 23c | Responses diagnostics: fallback OpenAI streaming adapter failures to local Ollama and add `AIIH_DEBUG_RESPONSES` trace logging |
| 2026-05-26 | Phase 23d | OpenAI router env bootstrap: load `.env` before settings for direct uvicorn starts |

## Phase 23 - Responses Client Compatibility (2026-05-25)
- [x] Accept bare Responses content parts such as `input_text` and plain `text` dictionaries as user input.
- [x] Emit Responses streaming text as `response.output_text.delta` for OpenAI-compatible clients.
- [x] Include assembled assistant text in `response.completed.output` so clients that render only the final event do not show blank replies.
- [x] Expose assembled assistant text as top-level `output_text` for Cherry Studio / ChatBox style Responses renderers.
- [x] Add compact `responses.trace` logging for input conversion, routing, provider completion, and response conversion.
- [x] Fallback `/v1/responses` streaming from unconfigured OpenAI provider to local Ollama instead of leaving the SSE client blank.
- [x] Load `.env` in `router/openai_router.py` before `settings` is initialized so `AIIH_DEBUG_RESPONSES` works outside the launcher.
- [x] Added regression coverage for bare input parts and streaming completed output.
