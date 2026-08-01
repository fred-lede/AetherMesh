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

## Phase 26 — Image Generation ✅ (2026-07-07)
### Task 1-2: Adapter
- [x] `providers/image_gen_adapter.py` — ImageGenAdapter with `generate(model, prompt, n)` via Ollama `/api/generate`
- [x] Fixed: `async def` + `await session.post()` on sync `requests.Session` → sync `def` (root cause of 500 error)
- [x] `tests/test_image_gen.py` — 3 adapter tests + 5 router tests (8 total)

### Task 3-6: Full Integration
- [x] `config/settings.py` — `image_gen_enabled`, `image_gen_default_model`, `image_gen_default_worker`
- [x] `config/cluster.yaml` — `node-mac-01: 192.168.1.100`
- [x] `.env` — `AIIH_IMAGE_GEN_ENABLED=true`
- [x] `.env.example` — image gen env var section
- [x] `config/models.yaml` — 4 image models (`x/z-image-turbo:fp8/bf16`, `x/flux2-klein:9b/4b`)
- [x] `runtime/orchestration/provider_router.py` — import guard + `_get_image_gen_adapter()` singleton with cooldown + ROUTE_PREFIXES
- [x] `router/image_router.py` — `POST /v1/images/generations` + `POST /v1/images/edits`
- [x] `router/openai_router.py` — conditional `include_router(image_router)`

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
- [x] 端對端測試驗證 `/v1/responses` with tools → tool execution → completed (2026-07-01)

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
| 2026-05-26 | Phase 23e | Responses role normalization: map `developer` input messages to Ollama-compatible `system` role |
| 2026-05-26 | Phase 23f | Responses SSE compatibility: emit stateful output item/content part/delta/done events with indexes |
| 2026-06-21 | Phase 23g | Responses circuit resilience: reroute streaming requests away from an open Ollama worker circuit |
| 2026-06-21 | Phase 23h | Chat streaming circuit resilience: apply the same pre-stream Ollama reroute for Codex clients |
| 2026-06-21 | Phase 23i | Codex tool ownership: return client function calls without AIIH executing noop handlers |
| 2026-06-29 | Phase 24 | Multi-Key Credential Pool: `providers/credential_pool.py` Composite Provider with automatic failover on 429/401/403, per-key cooldown tracking, transparent key rotation. 4 cloud adapters accept optional `api_key`/`base_url` params. Dashboard Cloud Credentials UI for add/remove keys. `config/credentials.json` config. SSE streaming hang fix: synchronous card rendering, async data loading. |

## Phase 23 - Responses Client Compatibility (2026-05-25)
- [x] Accept bare Responses content parts such as `input_text` and plain `text` dictionaries as user input.
- [x] Emit Responses streaming text as `response.output_text.delta` for OpenAI-compatible clients.
- [x] Include assembled assistant text in `response.completed.output` so clients that render only the final event do not show blank replies.
- [x] Expose assembled assistant text as top-level `output_text` for Cherry Studio / ChatBox style Responses renderers.
- [x] Add compact `responses.trace` logging for input conversion, routing, provider completion, and response conversion.
- [x] Fallback `/v1/responses` streaming from unconfigured OpenAI provider to local Ollama instead of leaving the SSE client blank.
- [x] Load `.env` in `router/openai_router.py` before `settings` is initialized so `AIIH_DEBUG_RESPONSES` works outside the launcher.
- [x] Normalize Responses `developer` messages to `system` before routing to Ollama-compatible providers.
- [x] Add `stream.failed` trace logging for Responses SSE provider exceptions.
- [x] Emit stateful Responses SSE events with stable `item_id`, `output_index`, and `content_index` fields.
- [x] Reroute Responses streams to an alternate capable Ollama worker when the selected worker's circuit is open.
- [x] Apply open-circuit Ollama rerouting to `/v1/chat/completions` streams used by Codex.
- [x] Return client-owned Responses function calls to Codex instead of executing temporary noop handlers in AIIH.
- [x] Added regression coverage for bare input parts and streaming completed output.

## Phase 25 — Local TTS + ASR (XTTS-v2 + faster-whisper) ✅ (2026-07-01)
### TTS (XTTS-v2)
- [x] `providers/tts_base.py` — ABC + TTSProviderError
- [x] `providers/xtts_adapter.py` — XTTSAdapter with voice CRUD, FP16, language auto-detect, TTS_HOME
- [x] `router/audio_router.py` — OpenAI-compatible `/v1/audio/speech`, `/v1/voices` (list/register/delete)
- [x] `provider_router.py` — xtts adapter factory + singleton
- [x] `config/settings.py` — tts_enabled, tts_model, tts_device, tts_models_dir
- [x] `config/models.yaml` — xtts-v2 on node-01:11435
- [x] 11 TTS tests (adapter + router)

### ASR (faster-whisper)
- [x] `providers/asr_base.py` — ABC + ASRProviderError
- [x] `providers/faster_whisper_adapter.py` — FasterWhisperAdapter with download_root
- [x] `router/audio_router.py` — `/v1/audio/transcriptions`, `/v1/audio/translations`
- [x] `provider_router.py` — asr adapter factory + singleton + ROUTE_PREFIXES
- [x] `config/settings.py` — asr_enabled, asr_model, asr_device, asr_compute_type, asr_models_dir
- [x] `config/models.yaml` — whisper-large-v3 on node-01:11435
- [x] 5 ASR endpoint tests + 10 adapter tests

### Error Handling Fix (2026-07-01)
- [x] TTS inference exceptions now wrapped in TTSProviderError (was raw 500)
- [x] OOM detection via class name (`OutOfMemoryError` → 503, others → 500)
- [x] ASR internal errors return 503 instead of 500
- [x] `register_voice` errors wrapped in TTSProviderError(422), auto-cleanup invalid dirs
- [x] `register_voice` router endpoint catches TTSProviderError → HTTPException
- [x] Fixed `TTSProviderError.message` → `str(e)` (RuntimeError uses args, not .message)
- [x] 2 new tests: `test_tts_inference_error_wrapped`, `test_tts_oom_error_returns_503`

### torchaudio/FFmpeg Compatibility Fix (2026-07-01)
- [x] `torchaudio.load()` monkey-patched with soundfile fallback (torchcodec needs FFmpeg shared DLLs)
- [x] `get_conditioning_latents` calls audio load via patched `torchaudio.load`
- [x] FFmpeg static build (`C:\mytools\ffmpeg`) works for FFmpeg output format conversion, but not for torchcodec DLL loading

### FP16 Conditioning Latent Fix (2026-07-01)
- [x] Root cause: `torch.autocast(dtype=float16)` on `get_conditioning_latents()` produces NaN in gpt_cond_latent → corrupted embedding → `torch.multinomial` assertion `input[0] != 0` crash
- [x] Fix: `register_voice()` uses autocast only inside `get_conditioning_latents()` for FP16 model compat, then converts output back to FP32 and replaces NaN with zeros
- [x] `_load_embedding()`: auto-converts FP32 latents to FP16 when `dtype="fp16"` for inference compat
- [x] Inference still uses autocast (model weights are FP16, latent arithmetic is fine)
- [x] Deleted corrupted voice `2982f022-e1ef-4220-97ac-495f3c77c4e5` (NaN in gpt_cond_latent from broken autocast)
- [x] Re-registered voice `075f432f` (TW-HsiaoChen) with FP32 latents → no NaN → TTS SUCCESS on CUDA
- [x] Fixed FP16 `register_voice` type mismatch: model is HalfTensor but `get_conditioning_latents` inputs are FloatTensor — now uses autocast + NaN cleanup + FP32 save
- [x] Also fixes: `transformers 4.57` `_sample` multinomial crash was symptom of NaN probs, not a transformers bug

### Verified Working
- [x] TTS with real voice: 200 OK, 48KB WAV audio returned
- [x] Voice registration: 200 OK with no-autocast fix
- [x] ASR endpoint: 200 OK (returns text)
- [x] Full TTS→ASR round-trip: TTS generates audio → ASR transcribes back correctly
- [x] CUDA TTS inference: `Hello, this is a test` → WAV shape (44544,), max 0.8174 — SUCCESS
- [x] All 41 audio tests passing

### Adapter Resilience (2026-07-01)
- [x] `provider_router.py`: TTS + ASR adapter factories now catch creation exceptions, return `None` instead of crashing
- [x] 30-second cooldown: failed adapter creation won't retry for 30s (prevents GPU hammering on broken state)
- [x] `audio_router.py`: `_resolve_adapter()` / `_resolve_asr_adapter()` return 503 when adapter is `None`
- [x] `_get_audio_duration()`: `sf.info()` fallback wrapped in try/catch (invalid WAV data → 0.0s instead of crash)
- [x] 4 new tests: `TestAdapterInitFailure` (2) + `TestProviderRouterResilience` (2)
- [x] Error responses now include detail messages (was empty 500 before)

### E2E Responses Tests (2026-07-01)
- [x] `tests/test_responses_e2e.py` — 6 tests covering full `/v1/responses` lifecycle
  - `TestResponsesE2EToolLoop` (3): router-level tests with mocked service/adapter
    - `test_with_tools_returns_function_call_to_client`: POST with tools → 200, status=completed
    - `test_no_tools_direct_completion`: POST without tools → 200, direct text response
    - `test_follow_up_with_function_call_output`: POST with prior function_call_output in input → completed with text
  - `TestResponsesToolLoopDirectExecution` (3): tool loop `run()` with real ToolExecutor
    - `test_run_single_tool_execution`: get_weather tool → executed → sunny in output
    - `test_run_multi_turn_tool_execution`: get_weather → calculate → 3 adapter calls
    - `test_run_unknown_tool_returns_error`: nonexistent tool → error result → model retries → completed

### Streaming WebSocket ASR (2026-07-03)
- [x] `providers/streaming_asr.py` — `StreamingASR` class: async ring buffer, energy VAD, sliding window Whisper via thread pool
- [x] `router/audio_router.py` — WebSocket endpoint at `/v1/audio/transcriptions/stream`, auth via `?api_key=` / `Authorization: Bearer`
- [x] WebSocket accept-before-auth fix (avoid HTTP 400)
- [x] Odd-length PCM guard in `add_audio()` and `_transcribe()`
- [x] No background task — inline `transcribe_if_ready()` with `wait_for` timeout loop
- [x] Auth bypass path in middleware for WS route
- [x] `tests/test_streaming_asr.py` — 12 tests (buffer, VAD, idle trigger, flush, concurrency)
- [x] `tests/test_ws_client.py` — sine-wave PCM streaming test script (supports `ws://`/`wss://`)
- [x] 白龍馬 protocol compatibility:
  - [x] Config frame `{"type":"config","provider":"aethermesh","lang":"zh"}` accepted as first message
  - [x] `lang` alias for `language` in config frames
  - [x] `is_final` field in transcript messages (interim=`false`, final=`true`)
  - [x] Interim results: window fill → `is_final: false`; idle timeout/flush → `is_final: true`

| 2026-07-07 | Phase 26 | ImageGenAdapter: Ollama `/api/generate` wrapper + 3 tests |
| 2026-07-07 | Phase 26 | Fixed: sync/async mismatch (root cause of 500), built full integration (router, settings, provider_router, models, cluster, tests) |
| 2026-07-07 | Phase 26 | Cherry Studio AI_RetryError root cause: TCP keep-alive timeout (5s default) kills pooled connections between 70s image gen requests. Fixes: `Connection: close` header + `--timeout-keep-alive 300` in launcher. |
| 2026-07-07 | Phase 26 | `/v1/images/edits` multipart/form-data support: Cherry Studio sends edits as form data with image file, not JSON. Handler now accepts `UploadFile` + `Form(...)`. All 8 tests passing. |
| 2026-07-07 | Phase 26 | ChatBox image gen mode fix: use built-in OpenAI provider (not custom provider) pointed to AetherMesh. No model alias needed. |
| 2026-07-07 | Phase 26 | macOS cluster node: `scripts/start-mac-node.sh`, launchd plist with `AIIH_CONTROL_IP`, node_agent + worker_agent register with Windows control plane. |

## Phase 27 — Custom OpenAI-Compatible Providers ✅ (2026-07-09)
- [x] `config/custom_providers.json` — gitignored JSON store for name→(api_type, base_url, api_key) configs
- [x] `config/settings.py` — `load_custom_providers()` + `save_custom_providers()` helpers
- [x] `runtime/orchestration/provider_router.py` — `_CUSTOM_PROVIDERS` cache, `_load_custom_providers()` filter, `reload_custom_providers()`, `custom_provider_status()`, adapter() fallback
- [x] `runtime/orchestration/routing_engine.py` — `register_custom_providers()` / `unregister_custom_providers()` with built-in provider protection, `_check_provider_credentials()` reads custom configs, `_cloud_adapter_worker()` handles custom providers
- [x] `dashboard/dashboard_server.py` — CRUD at `/api/custom-providers`, probe at `/{name}/probe`, reload at `/reload`, overview SSE includes custom_providers
- [x] `dashboard/templates/index.html` — Custom Providers section in Providers tab
- [x] `dashboard/static/dashboard.js` — `renderCustomProviders()`, `addCustomProvider()`, `probeCustomProvider()`, `deleteCustomProvider()`, `editCustomProvider()`
- [x] `tests/test_custom_providers.py` — 28 tests: Settings load/save, _load_custom_providers filtering, adapter resolution, reload, register/unregister, Dashboard API CRUD
- [x] **Post-deployment fix (2026-07-09)**: 3 routing fixes for custom provider model resolution
  - Startup registration: `_CUSTOM_PROVIDERS` syncs to routing engine at module load (provider_router.py:136)
  - Prefix fallback: `provider_for_model()`/`resolve_provider()` match `agnes-2.0-flash` → `agnes` by prefix (no models.yaml entry required)
  - Dispatch bypass: `_resolve_provider_and_worker()` treats custom providers like cloud providers — returns immediately without worker dispatch |

## Phase 28 — OpenAI Function Tool Schema 標準化 ✅ (2026-08-01)
- [x] 新增 `ensure_parameters_schema()` helper (`runtime/tools/tool_registry.py`)：缺參數/空/非法 schema 一律回傳 `{"type":"object","properties":{},"additionalProperties":false}`
- [x] `ToolRegistry.get_openai_tools()` 對空 `input_schema` 的工具回傳有效 JSON Schema
- [x] `openai_handler._ensure_openai_tools()` 修復兩處缺 `parameters` 的 bug：
  - flat 格式 `{"type":"function","name":...}`（無 function 鍵）現在補上 `parameters`
  - nested 格式 `{"type":"function","function":{...}}` 缺 `parameters` 時補上（原直接 pass-through）
  - plugin/integration 分支改用 helper（原預設無 `additionalProperties`）
- [x] `anthropic_converter._anthropic_tools_to_openai()` 修復 `input_schema={}` → `parameters:{}` 的無效 schema（改為 helper 預設）
- [x] `tool_loop._register_temp_tools()` 改用 helper（原本就有預設，統一風格）
- [x] `tests/test_tool_schema.py` — 14 tests（helper 3 + registry 2 + _ensure_openai_tools 6 + anthropic 3）
- [x] 驗證：相關套件 120 passed, 6 skipped（Linux-only sandbox）；`test_image_gen::test_edits_requires_prompt` 為既有 422vs400 環境失敗，與本次無關

## Phase 29 — Codex Namespace 工具協議支援 ✅ (2026-08-01)
- [x] 盤點 Codex app-server 官方 README + `responses_api.rs` + `dynamic_tools_tests.rs`：namespace 是新的 Codex 工具協議一部分
  - namespace 結構 `{"type":"namespace","name","description","tools":[{"type":"function",...}]}`
  - namespace 內 function 用 flat 格式（`name`+`inputSchema`，非 OpenAI nested `function` dict）
  - namespace name 須 `^[a-zA-Z0-9_-]+$` 1–64 字元；保留字：functions、multi_tool_use、file_search、web、browser、image_gen、computer、container、terminal、python、python_user_visible、api_tool、tool_search、submodel_delegator
  - `deferLoading` flag（預設 false，僅 namespace 內可用）；呼叫用分開的 `namespace`+`name` 欄位
- [x] `openai_handler._ensure_openai_tools()` 新增 namespace 分支：unwrapped 子 function，逐一套 `ensure_parameters_schema`，限定名 `{namespace}.{name}`；同時支援 flat（`inputSchema`）與 nested（`function` dict）格式；`strict` 透傳
- [x] `ollama_adapter._tools_for_ollama()` namespace 分支修復：原本只認 nested `function` dict，Codex flat 格式會**靜默丟棄**；現在支援 flat + nested，並限定名 `{namespace}.{name}`
- [x] `input_converter._parse_input_item()` function_call 帶 `namespace` 欄位時限定為 `{namespace}.{name}`（避免重複限定）
- [x] 新增測試：`test_tool_schema.py` +5（namespace flat/nested/缺 schema/無 name）、`test_ollama_adapter.py` +2（flat + nested 限定名）、`test_responses_tool_loop.py` +2（namespace 限定 + 不重複限定）
- [x] 驗證：相關套件 68 passed；全量 466 passed, 6 skipped（5 個環境既有失敗：test_file_parser 缺 pypdf/python-docx/pptx、test_token_counting 指標——stash 確認與本次無關）

## Phase 30 — NVIDIA NIM web_search 工具型別修正 ✅ (2026-08-01)
- [x] Bug：Codex 送 `{"type":"web_search"}` 工具 → 路由到 NVIDIA NIM → NIM 報 `unknown variant web_search, expected function` 拒絕請求
- [x] Root cause：server tool policy 只在 Anthropic path（`messages_adapter.py`）執行；OpenAI path 的 `_ensure_openai_tools` 對 OpenAI 原生 Responses API 正確透傳 `web_search`，但 NIM `/chat/completions` 只接受 `type:"function"`。Ollama/Gemini adapter 都在 adapter 層過濾非 function 工具，唯 NIM 直接轉發
- [x] 修正：`nvidia_nim_adapter._filter_tools()` 過濾掉非 `type:"function"` 的工具；`chat()`/`stream()` 都套用
- [x] `tests/test_nvidia_nim_adapter.py` — 新增 3 tests（chat 丟棄 web_search、stream 丟棄、無 tools 時不變）
- [x] 驗證：相關套件 93 passed

## Phase 31 — NVIDIA NIM Responses-only 參數過濾 ✅ (2026-08-01)
- [x] Bug：Codex streaming `/v1/responses` → NIM 報 `Validation: Unsupported parameter(s): include, reasoning, client_metadata, prompt_cache_key`
- [x] Root cause：`handle_streaming_responses` 把 Responses payload 複製成 `chat_payload` 後只 pop `input/instructions/previous_response_id/store/stream`，`include`/`reasoning`/`client_metadata`/`prompt_cache_key` 等 Responses-only 參數原封不動送進 NIM `/chat/completions`；Gemini/Ollama 用白名單建 body 所以免疫，NIM 直接轉發
- [x] 修正：`nvidia_nim_adapter._chat_payload()` 改為白名單制（`_CHAT_KEYS` 只含 chat-completions 參數），保留 `_filter_tools` 功能（只送 `type:"function"`）；`chat()`/`stream()` 都走此路徑；NIM 只會走 `chat()`/`stream()`（`responses()` 僅 OpenAI provider 使用）
- [x] `tests/test_nvidia_nim_adapter.py` — 新增 2 tests（chat/stream 剝離 Responses-only 參數、保留合法 chat 參數）
- [x] 驗證：相關套件 106 passed

## Phase 31b — NVIDIA NIM Namespaced Tool Name 消毒 ✅ (2026-08-01)
- [x] Bug：Codex streaming + NIM 報 `Validation: Function at index 8 has an invalid name: "mcp__codegraph.codegraph_explore". Only a-z, A-Z, 0-9, underscores, and dashes are allowed.`
- [x] Root cause：`_ensure_openai_tools` 把 namespace 工具限定成 `{namespace}.{name}`（Codex MCP namespace `mcp__codegraph` + function `codegraph_explore` → `mcp__codegraph.codegraph_explore`）。NIM function name 只允許 `[a-zA-Z0-9_-]`，dot 在 validation 被拒
- [x] 修正（完全限定在 NIM adapter；adapter 每個 request 都是新 instance，所以 per-request map 安全）：
  - `NvidiaNIMAdapter._sanitize_tool_name()` — `.` → `__`（符合 `mcp__server__tool` 慣例），其他非法字元 → `_`
  - `_chat_payload()` 記錄 `self._tool_name_map`（sanitized → original），並複製 fn dict（不污染 input，retry/fallback 時 idempotent）
  - `chat()` 呼叫 `_restore_completion_names()`、`stream()` 呼叫 `_restore_chunk_names()`，讓 client (Codex) 收到原本的 dotted `namespace.name` 以便 round-trip dispatch
  - 確認兩條 passthrough 路徑：streaming 直接 proxy chunks 給 client（`openai_handler.py:1032`）、non-streaming `run_with_client_tools` 只做單次 `adapter.chat()` — 都不在 AetherMesh 執行，reverse-map 完全在 adapter 內完成
- [x] `tests/test_nvidia_nim_adapter.py` — +4 tests：chat sanitize+restore、stream sanitize+restore、sanitize 字元保留
- [x] 驗證：相關套件 150 passed

## Phase 32 — OpenAI Passthrough 工具參數正規化 ✅ (2026-08-01)
- [x] Bug：streaming `/v1/responses` → 上游 OpenAI-compatible（Rust serde）報 `Invalid JSON data: Failed to deserialize the JSON body into the target type: tools[139].function: missing field parameters`（400 json_parse_error）
- [x] Root cause：`provider == "openai"` 的兩條 passthrough 路徑把未正規化的工具原封不動轉發：
  - streaming：`openai_payload = dict(payload)`（raw client payload，無 `messages`、tools 沒跑 `_ensure_openai_tools`）→ `openai_adapter.stream()` 送 `/chat/completions`
  - non-streaming：`adapter.responses(original_payload)`（raw payload，tools 沒跑 `_ensure_openai_tools`）
  - 結果：flat/namespace 工具的 `function` 缺 `parameters`，嚴格上游 reject（其他 provider 走 `effective_payload` 所以免疫）
- [x] 修正（`openai_handler.py`）：
  - streaming 分支改用 `outer_state["payload"]`（即已正規化的 `effective_payload`，含 messages + tools + 移除 responses-only 參數）
  - non-streaming 分支在 `responses(original_payload)` 前對 `original_payload["tools"]` 跑 `_ensure_openai_tools`
- [x] `tests/test_responses_e2e.py` — +2 tests：streaming/non-streaming openai passthrough 的 tools 皆含 `parameters`、namespace 扁平化、streaming 有 `messages`
- [x] 驗證：相關套件 110 passed

## Phase 32b — tool_choice 在無 tools 時剝離 ✅ (2026-08-01)
- [x] Bug：上游 OpenAI-compatible（Rust serde）報 `__all__: Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.`（400 invalid_request_error）
- [x] Root cause：client 送 `tool_choice` 但無 tools（或 tools 在 `_ensure_openai_tools` 正規化後變空，例如空 namespace），strict 上游在沒有 tools 時不允許 `tool_choice`
- [x] 修正（三處防線）：
  - `openai_handler._normalize_payload_for_provider()` — tools 正規化後若 `tools` 為空/缺，`pop("tool_choice")`（覆蓋 streaming openai + 所有非 openai provider 路徑）
  - `openai_handler.handle_responses()` non-streaming openai 分支 — 對 `original_payload` 做同樣剝離
  - `nvidia_nim_adapter._chat_payload()` — 建 body 後若 `tools` 為空/缺，`pop("tool_choice")`（覆蓋 NIM chat 路徑）
- [x] `tests/test_responses_e2e.py` — +2 tests：streaming/non-streaming openai passthrough 在無 tools 時不帶 `tool_choice`
- [x] 驗證：`test_responses_e2e.py` 10 passed；`test_orchestration.py` + `test_nvidia_nim_adapter.py` 24 passed

## Phase 32c — 工具參數防線強化 + 非同步路徑正規化 ✅ (2026-08-01)
- [x] 現象：Phase 32/32b 之後相同錯誤 `tools[139].function: missing field parameters`（400 json_parse_error）再次出現
- [x] 調查結論：目前程式碼所有 Codex 使用路徑都已正規化（/v1/responses ± streaming、/v1/chat/completions ± streaming、/v1/messages 皆確保 parameters），相同錯誤重現最可能是**伺服器跑舊版程式（未 restart，早於 Phase 32）**
- [x] 但仍補強兩處真實缺口（defense-in-depth，同類錯誤）：
  - `_ensure_openai_tools()` — 新增最終保證 pass：所有輸出的 `{"type":"function","function":{...}}` 必定含 `parameters`（封閉 plugin/integration 子工具 raw append 缺 parameters 的漏洞）
  - `_enqueue_async_task()` — 送 control plane 前對 `tools` 跑 `_ensure_openai_tools`（封閉非同步 worker 送 raw payload 的漏洞）
- [x] `tests/test_responses_e2e.py` — +1 test：plugin 子工具缺 parameters → 送出時補上 default schema
- [x] 驗證：`test_responses_e2e.py` + `test_orchestration.py` 27 passed
- [x] ⚠️ 使用者需 restart AetherMesh server，否則舊 process 仍以未正規化 payload 轉發

## Phase 32d — openai passthrough 只轉發 function 工具 ✅ (2026-08-01)
- [x] 根因確認（`stream.failed` trace）：正規化後 index 144 是 `{"type": "web_search", ...}`（無 `function` 欄位）。嚴格上游 gateway（Rust serde，chat-completions 純 function-tools）不支援 `web_search` tool type，解析時報 `tools[144].function: missing field parameters`（400 json_parse_error）
- [x] 修正：`_filter_openai_tools()` — 只保留 `type == "function"` 的工具；套用於：
  - `_normalize_payload_for_provider()`（provider == "openai"，覆蓋 streaming responses + chat ± streaming）
  - `handle_responses()` non-streaming openai 分支（`original_payload`）
  - NIM 不受影響（filter 只在 provider == "openai"；Phase 30 已支援 NIM web_search）
- [x] `tests/test_responses_e2e.py` — +2 tests：streaming/non-streaming openai 丟棄 web_search、保留 function 工具
- [x] 驗證：`test_responses_e2e.py` + `test_orchestration.py` + `test_nvidia_nim_adapter.py` 37 passed

## Phase 32e — custom provider（agnes）也套用 function-only 過濾 ✅ (2026-08-01)
- [x] 現象：Phase 32d 之後 trace 仍報 `tools[144].function: missing field parameters`（400 json_parse_error）
- [x] 根因：`stream.failed` trace 顯示 `"provider": "agnes"`（非 "openai"）。自訂 provider 一律用 `OpenAIAdapter`（`provider_router.py:149` `_CLOUD_ADAPTERS[name] = OpenAIAdapter`），但 `_filter_openai_tools` 只在 `provider == "openai"` 套用 → agnes 繞過過濾
- [x] 修正（`openai_handler.py`）：
  - `_normalize_payload_for_provider()`：條件改為 `provider == "openai" or is_custom_provider(provider)`（覆蓋 streaming responses）
  - `handle_responses()` non-streaming 工具迴圈路徑：`tools` 傳入 `responses_tool_loop` 前也過濾（`tool_loop.py:186` 會用 raw `tools` 覆寫 `chat_payload["tools"]`，否則 web_search 仍會送出）
- [x] `tests/test_responses_e2e.py` — +2 tests：streaming/non-streaming custom provider（agnes）丟棄 web_search、保留 function 工具
- [x] 驗證：三套件 39 passed

## Phase 32f — SSE 串流 StopIteration RuntimeError（Python 3.14）✅ (2026-08-01)
- [x] 現象：重啟載入 32e 後，web_search 過濾已生效（trace 顯示 route_selected 139 tools、無 web_search），但 streaming 結束時 server 崩潰：`RuntimeError: StopIteration interacts badly with generators and cannot be raised into a Future`（`router/streaming_router.py:51`）
- [x] 根因：`async_stream_response()` 用 `loop.run_in_executor(None, next, iter_obj)` 推進 generator。generator 耗盡時 `next` 在 worker thread 內拋 `StopIteration`；Python 3.14 不允許 StopIteration 傳入 Future → `except StopIteration` 永遠接不到，轉成 RuntimeError
- [x] 修正（`router/streaming_router.py`）：改用 `partial(next, iter_obj, _SENTINEL)` + sentinel 比對結束，完全不拋 StopIteration
- [x] `tests/test_streaming_router.py` — 新增（format_sse_event、stream_response、async_stream_response 耗盡/空迭代）
- [x] 驗證：streaming_router + responses_e2e 20 passed
