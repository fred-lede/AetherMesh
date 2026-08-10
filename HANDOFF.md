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
