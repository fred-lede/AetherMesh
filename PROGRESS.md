# Progress

## 2026-07-09 — Post-deployment fixes: probe, enable-state, credential cards

### Fixes
- **Probe**: `_probe_provider()` raised 404 for custom providers — added `is_custom_provider()` check to delegate to `_probe_custom_provider()`
- **Route disabled on startup**: `register_custom_providers()` now calls `routing_engine.set_provider_enabled(name, True)`, and `_load_state()` no longer filters out unknown providers (preserves prior manual toggle across restarts)
- **Cloud Credentials cards**: template + JS now dynamically include custom providers; `renderCredentials()` injects/removes credential cards when providers are added/deleted without page refresh
- **Test state leakage**: module-level cleanup deletes stale `routing_state.yaml` before it's loaded; teardown restores `_provider_enabled`

## 2026-08-01 — Phase 32b: strip `tool_choice` when no tools present

### Fixes
- Strict OpenAI-compatible upstream (Rust serde) rejected `tool_choice` when `tools` absent/empty: `__all__: Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.`
- `openai_handler._normalize_payload_for_provider()` now pops `tool_choice` when `tools` ends up empty/missing after `_ensure_openai_tools` (covers streaming openai + all non-openai provider paths)
- `openai_handler.handle_responses()` non-streaming openai branch pops `tool_choice` on `original_payload` when tools empty/missing
- `nvidia_nim_adapter._chat_payload()` pops `tool_choice` when `tools` empty/missing (covers NIM chat path)
- Tests: +2 in `tests/test_responses_e2e.py` (streaming/non-streaming openai passthrough drop `tool_choice` without tools) — 10 passed; orchestration + NIM adapter suites 24 passed

## 2026-08-01 — Phase 32e: function-only tool filter covers custom providers

### Fixes
- `stream.failed` trace showed provider `agnes` (custom) still forwarding `web_search` → `tools[144].function: missing field parameters` (400 json_parse_error)
- Root cause: custom providers use `OpenAIAdapter` (`provider_router.py` registers `_CLOUD_ADAPTERS[name] = OpenAIAdapter`), but `_filter_openai_tools` guard only matched `provider == "openai"`
- `_normalize_payload_for_provider()` guard extended to `provider == "openai" or is_custom_provider(provider)`
- `handle_responses()` non-streaming tool-loop path: `tools` now filtered before `responses_tool_loop` (loop overwrites `chat_payload["tools"]` with the raw client list, re-introducing `web_search`)
- Tests: +2 in `tests/test_responses_e2e.py` (streaming/non-streaming custom provider drops `web_search`, keeps function) — 39 passed across e2e + orchestration + NIM suites

## 2026-08-01 — Phase 32f: SSE stream StopIteration RuntimeError on Python 3.14

### Fixes
- After 32e restart, `web_search` filter confirmed working (`route_selected` = 139 tools, no `web_search`), but streaming requests crashed at completion with `RuntimeError: StopIteration interacts badly with generators and cannot be raised into a Future` (`router/streaming_router.py:51`)
- Root cause: `async_stream_response()` advanced the generator with `loop.run_in_executor(None, next, iter_obj)`; on exhaustion `next()` raises `StopIteration` in the worker thread, which Python 3.14 converts to `RuntimeError`, bypassing the `except StopIteration` guard
- Fix: `partial(next, iter_obj, _SENTINEL)` + sentinel comparison — loop ends without raising `StopIteration`
- Tests: new `tests/test_streaming_router.py` (format_sse_event, sync stream, async stream exhaustion, empty iterator) — 20 passed

## 2026-08-01 — Phase 32g: upstream 400 "No user query found in messages"

### Fixes
- After 32e/32f, Codex first request succeeded (200, 139 function tools), but the tool-loop follow-up failed with `BadRequestError: No user query found in messages` (400) from the agnes upstream gateway
- Root cause: `runtime/responses/input_converter.py:_truncate_input_list()` (`truncation:"auto"`) keeps only the last 8 non-system messages. When a single turn is `[user, assistant(tool_calls), tool×7]` (9 non-system), the user message falls outside the window and is dropped, leaving `[assistant(tool_calls), tool×7]` → upstream rejects with no user query
- Fix: after slicing, if the kept window contains no `role: user`, move the cut forward to the last user message in `non_system`
- Tests: new `tests/test_input_converter.py` (4 tests) — 68 passed across input_converter + responses e2e/tool_loop/adapter + streaming_router suites

## 2026-08-11 — Phase 34: Structured Output / Batch API / Audit Query / Realtime API

### Structured Outputs (`response_format`)
- New `runtime/orchestration/structured_output.py`: `response_format_schema()` parses `json_schema`/`json_object` modes; `extract_json_content()` handles code fences + embedded JSON; `validate_json()` recursively checks type/required/properties/items/enum/min-max; `build_repair_messages()` keeps conversation history + injects schema; `apply_structured_output()` strips `response_format` before the adapter call, validates, and runs a bounded repair loop (max 2 retries) then normalizes valid JSON back into `choices[0].message.content`
- Wired into `handle_chat` non-streaming path; `StructuredOutputError` → HTTPException 422 in `router/openai/chat_adapter.py`
- Tests: `tests/test_structured_output.py` (20) — all pass

### Batch API (`/v1/batches`)
- New `runtime/orchestration/batch_manager.py`: JSONL input validation (custom_id/method/url/body), background-thread processing over `handle_chat`/`handle_responses`/`handle_embeddings`, OpenAI-format output JSONL (`file_batch_out_*`), per-request error capture, JSON persistence + reload, cancel support
- New `router/openai/batches_adapter.py`: `POST/GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`
- Tests: `tests/test_batch_manager.py` (12) — all pass

### Audit Log Query API (`/v1/audit/logs`)
- `AuditLog.query()` now supports action/actor/time-range/details filter + offset/limit; `recent_events` keeps "latest N" semantics
- New `router/audit_router.py`: unified query over `security` + `routing` JSONL sources, timestamp desc, `has_more`; `GET /v1/audit/sources`
- Tests: `tests/test_audit_log_query.py` (11) — all pass

### Realtime API (`/v1/realtime` WebSocket)
- New `runtime/realtime/realtime_session.py`: `RealtimeSession` state machine (session.update / conversation.item.create / ping / audio rejection) + `build_messages()` (instructions → system, items → user/assistant messages)
- New `router/realtime_router.py`: WS endpoint with Bearer/`?api_key=` auth, `session.created`, `response.create` → `asyncio.to_thread(handle_chat)` → `response.created` / `output_item.added` / `content_part.added` / chunked `text.delta` / `output_item.done` / `response.done` (failures → `status: failed`)
- Tests: `tests/test_realtime_session.py` (12) — all pass

### Verification
- Related suites 70 passed (structured_output 20 + batch 12 + audit 11 + realtime 12 + orchestration 15); `router.openai_router` imports cleanly with all new routers wired
