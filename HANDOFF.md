# Handoff

## 2026-09-22 — Context-length feature + opencode/Telegram display integration

State: feature complete, all commits pushed.

### Added this session (commits already pushed)
- **Model-level context length** (`4c72b6a`, `aaa0273`): `runtime/orchestration/model_context.py` + `model_context_cli.py`, per-model ctx resolution (models.yaml → auto-fetch cache → provider fallback), wired into `provider_scoring.py`/`execution_selector.py`.
- **`/v1/models` fresh read + metadata** (`57d564b`): `openai_handler.py`/`anthropic_converter.py` `list_models` reads `config/models.yaml` per request and exposes `context_length`/`context_window`.
- **Supervisor stale-sentry fix** (`5e15566`): `_read_sentry()` ignores sentry files older than 60s so a conflicting independent process on a sentry port can't block scheduled restart.
- **models.yaml ctx fill** (`199546a`): 46 models carry `context_length`.

### Client-side finding (no AetherMesh code needed)
- Telegram's 200k came from `@grinev/opencode-telegram-bot`'s `model-context-limit-service.js`: it reads `limit.context` from the opencode server API and falls back to `200000` when absent. AetherMesh `/v1/models` is never consulted by the bot.
- Resolved by adding `limit.context/output` to the models in `~/.config/opencode/opencode.jsonc` (aiih/aiih2 providers). Verified: Telegram now shows 1M.

### Notes
- NVIDIA NIM `/v1/models` returns no context fields (confirmed with live key) — the "1M via nvidia_nim" impression comes from opencode's models.dev DB, not the API.
- AetherMesh's direct-API context metadata still serves GUI/dashboard clients; keep it.

## 2026-08-11 — Phase 34: OpenAI-compatible surface expansion

State: all features implemented and tested; **not yet deployed** (server restart required).

### Added this session
- **Structured Outputs**: `runtime/orchestration/structured_output.py` — `response_format` schema validation + bounded repair loop wired into non-streaming `handle_chat`; `StructuredOutputError` → HTTP 422.
- **Batch API**: `runtime/orchestration/batch_manager.py` + `router/openai/batches_adapter.py` — `/v1/batches` CRUD, background JSONL processing over chat/responses/embeddings, persisted state, cancel.
- **Audit Query API**: `runtime/security/audit_log.py` `query()` + `router/audit_router.py` — unified `/v1/audit/logs` over security + routing JSONL.
- **Realtime API**: `runtime/realtime/realtime_session.py` + `router/realtime_router.py` — text-oriented WS `/v1/realtime` (session.update / conversation.item.create / response.create with delta streaming).
- Docs: README endpoints, PROGRESS.md appended, TASK.md Phase 34.

### Remaining / notes
- Phase 33 uncommitted stability fixes (event bridge loop, routing_state.yaml WinError 32 retry, token usage COUNT aggregation, per-key token usage in dashboard) are included in the same commit series — verify after deploy.
- Streaming `handle_chat`/`handle_streaming_chat` do **not** apply structured-output validation (non-streaming only).
- Realtime API is text-only; audio input events are rejected with an `error` event.
- Batch requests run in daemon threads; output files land in `settings.upload_dir` as `file_batch_out_*`.

### Open debt (from Phase 33 audit, unresolved)
- Unbounded in-memory structures: episodic/semantic memory, metrics histograms, tracer spans, rate-limiter buckets, session/worker registries, event-bus history pop(0) O(n).

## 2026-08-23 — Watchdog 告警子系統（Phase 43）
- 架構：launcher process 內 Watchdog thread（runtime/health/watchdog.py）→ AlertManager（runtime/alerting/）→ Telegram / Synology Chat
- 設定檔：config/notifications.json（channels: telegram/synology_chat；watchdog: interval/rules/auto_restart）。Dashboard System tab 可線上編輯+測試，mtime 熱重載免重啟
- 當機偵測原理：/health 探測 timeout = event loop 卡死前兆；連續 N 次失敗即告警，持續逾 restart_after_s 自動重啟（有 cooldown + 每日上限防護）
- 已知待辦：Telegram bot token / Synology webhook 由使用者自行建立後於 Dashboard 填入；test_custom_providers 的 11 個 401 為既有跨測試 auth 污染（乾淨 codebase 復現），污染源待排查

## 2026-08-24 補充 — Service Control
- config/services.json = 服務期望狀態（gitignored，範本 services.json.example）。Dashboard Service Control 卡片寫入 → launcher 每 1s reconcile（stop/start）→ watchdog 跳過 intentionally_stopped 服務。watchdog 另有 startup_grace_s=180 防啟動期誤報、exclude_services 永久排除清單。注意：openai_router(8001) 是 OpenAI 相容總入口，停用會斷所有 OpenAI 格式 API。

## 2026-08-28 — 整棧自癒機制（重要）
- 之前『跑一天就卡』根因：launcher 前景執行 + AIIH-Platform 排程 ExecutionTimeLimit=PT72H + 0xC000013A 外部終止 → 整棧靜默死亡。in-process watchdog 隨 launcher 一起死，無法自癒。
- 現在：獨立 supervisor（python -m runtime.launcher supervise）常駐監控，整棧死即重拉；launcher 寫 runtime/launcher/launcher.pid + launcher_sentry.json 供判活。AIIH-Platform 排程已改為執行 scripts/start_supervisor.bat、ExecutionTimeLimit=PT0S、S4U logon、restart-on-failure。
- 注意：AIIH-Platform 排程曾以 Password logon 記錄 fred 憑證；本次改 S4U（不需密碼）。若排程任務無法以 S4U 於開機後正確啟動（需檔案權限），可能需手動重填密碼：schtasks /Change /TN AIIH-Platform /RU fred /RP <password>。
