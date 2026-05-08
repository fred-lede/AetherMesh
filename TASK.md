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

## Phase 8 — Provider Capability Registry 🟡
- [x] providers/registry.py (Capability enum + entry dataclass)
- [ ] ProviderCapabilityRegistry class (register/get_providers_for/score_provider)
- [ ] Extended scoring: GPU pressure, cost, tool requirements

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

## Phase 13 — Router Simplification ❌
- [ ] router/openai/ (chat_adapter, responses_adapter, models_adapter, embeddings_adapter, rerank_adapter)
- [ ] router/anthropic/ (messages_adapter)
- [ ] router/mcp/ (mcp_adapter)

## Phase 14 — Clean Architecture ❌
- [ ] cli/ (aethermesh_cli.py)
- [ ] clients/ (openai_sdk, anthropic_sdk, mcp_sdk)
- [ ] protocols/ 補完 (openai/, mcp/)

## Phase 15 — Documentation 🟡
- [x] docs/architecture/overview.md
- [x] docs/architecture/runtime-lifecycle.md
- [x] docs/runtime/tool-lifecycle.md
- [x] docs/mcp/gateway-architecture.md
- [x] docs/providers/capability-registry.md
- [x] docs/gpu/scheduling.md
- [x] docs/security/policies.md
- [ ] docs/runtime/agent-lifecycle.md
- [ ] docs/runtime/session-lifecycle.md
- [ ] docs/mcp/bridge-pattern.md
- [ ] docs/tools/builtin-tools.md
- [ ] docs/tools/tool-policy.md
- [ ] docs/tools/web-search.md
- [ ] docs/providers/adding-new-provider.md
- [ ] docs/gpu/topology.md
- [ ] docs/security/sandbox.md

---

## 執行記錄

| Date | Phase | Action |
|---|---|---|
| 2026-05-07 | All | 15-phase refactoring 完成 (平台更名 + runtime/ + protocols/) |
| 2026-05-07 | Phase 2/3 | Dashboard Jinja2 crash fix (pre-compiled templates) |
| 2026-05-07 | Phase 2/3 | Windows Dashboard 驗證通過 |
