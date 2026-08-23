# Handoff

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
