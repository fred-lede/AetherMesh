# Progress

## 2026-07-09 â€” Post-deployment fixes: probe, enable-state, credential cards

### Fixes
- **Probe**: `_probe_provider()` raised 404 for custom providers â€” added `is_custom_provider()` check to delegate to `_probe_custom_provider()`
- **Route disabled on startup**: `register_custom_providers()` now calls `routing_engine.set_provider_enabled(name, True)`, and `_load_state()` no longer filters out unknown providers (preserves prior manual toggle across restarts)
- **Cloud Credentials cards**: template + JS now dynamically include custom providers; `renderCredentials()` injects/removes credential cards when providers are added/deleted without page refresh
- **Test state leakage**: module-level cleanup deletes stale `routing_state.yaml` before it's loaded; teardown restores `_provider_enabled`

## 2026-08-01 â€” Phase 32b: strip `tool_choice` when no tools present

### Fixes
- Strict OpenAI-compatible upstream (Rust serde) rejected `tool_choice` when `tools` absent/empty: `__all__: Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.`
- `openai_handler._normalize_payload_for_provider()` now pops `tool_choice` when `tools` ends up empty/missing after `_ensure_openai_tools` (covers streaming openai + all non-openai provider paths)
- `openai_handler.handle_responses()` non-streaming openai branch pops `tool_choice` on `original_payload` when tools empty/missing
- `nvidia_nim_adapter._chat_payload()` pops `tool_choice` when `tools` empty/missing (covers NIM chat path)
- Tests: +2 in `tests/test_responses_e2e.py` (streaming/non-streaming openai passthrough drop `tool_choice` without tools) â€” 10 passed; orchestration + NIM adapter suites 24 passed

## 2026-08-01 â€” Phase 32e: function-only tool filter covers custom providers

### Fixes
- `stream.failed` trace showed provider `agnes` (custom) still forwarding `web_search` â†’ `tools[144].function: missing field parameters` (400 json_parse_error)
- Root cause: custom providers use `OpenAIAdapter` (`provider_router.py` registers `_CLOUD_ADAPTERS[name] = OpenAIAdapter`), but `_filter_openai_tools` guard only matched `provider == "openai"`
- `_normalize_payload_for_provider()` guard extended to `provider == "openai" or is_custom_provider(provider)`
- `handle_responses()` non-streaming tool-loop path: `tools` now filtered before `responses_tool_loop` (loop overwrites `chat_payload["tools"]` with the raw client list, re-introducing `web_search`)
- Tests: +2 in `tests/test_responses_e2e.py` (streaming/non-streaming custom provider drops `web_search`, keeps function) â€” 39 passed across e2e + orchestration + NIM suites

## 2026-08-01 â€” Phase 32f: SSE stream StopIteration RuntimeError on Python 3.14

### Fixes
- After 32e restart, `web_search` filter confirmed working (`route_selected` = 139 tools, no `web_search`), but streaming requests crashed at completion with `RuntimeError: StopIteration interacts badly with generators and cannot be raised into a Future` (`router/streaming_router.py:51`)
- Root cause: `async_stream_response()` advanced the generator with `loop.run_in_executor(None, next, iter_obj)`; on exhaustion `next()` raises `StopIteration` in the worker thread, which Python 3.14 converts to `RuntimeError`, bypassing the `except StopIteration` guard
- Fix: `partial(next, iter_obj, _SENTINEL)` + sentinel comparison â€” loop ends without raising `StopIteration`
- Tests: new `tests/test_streaming_router.py` (format_sse_event, sync stream, async stream exhaustion, empty iterator) â€” 20 passed

## 2026-08-01 â€” Phase 32g: upstream 400 "No user query found in messages"

### Fixes
- After 32e/32f, Codex first request succeeded (200, 139 function tools), but the tool-loop follow-up failed with `BadRequestError: No user query found in messages` (400) from the agnes upstream gateway
- Root cause: `runtime/responses/input_converter.py:_truncate_input_list()` (`truncation:"auto"`) keeps only the last 8 non-system messages. When a single turn is `[user, assistant(tool_calls), toolÃ—7]` (9 non-system), the user message falls outside the window and is dropped, leaving `[assistant(tool_calls), toolÃ—7]` â†’ upstream rejects with no user query
- Fix: after slicing, if the kept window contains no `role: user`, move the cut forward to the last user message in `non_system`
- Tests: new `tests/test_input_converter.py` (4 tests) â€” 68 passed across input_converter + responses e2e/tool_loop/adapter + streaming_router suites

## 2026-08-11 â€” Phase 34: Structured Output / Batch API / Audit Query / Realtime API

### Structured Outputs (`response_format`)
- New `runtime/orchestration/structured_output.py`: `response_format_schema()` parses `json_schema`/`json_object` modes; `extract_json_content()` handles code fences + embedded JSON; `validate_json()` recursively checks type/required/properties/items/enum/min-max; `build_repair_messages()` keeps conversation history + injects schema; `apply_structured_output()` strips `response_format` before the adapter call, validates, and runs a bounded repair loop (max 2 retries) then normalizes valid JSON back into `choices[0].message.content`
- Wired into `handle_chat` non-streaming path; `StructuredOutputError` â†’ HTTPException 422 in `router/openai/chat_adapter.py`
- Tests: `tests/test_structured_output.py` (20) â€” all pass

### Batch API (`/v1/batches`)
- New `runtime/orchestration/batch_manager.py`: JSONL input validation (custom_id/method/url/body), background-thread processing over `handle_chat`/`handle_responses`/`handle_embeddings`, OpenAI-format output JSONL (`file_batch_out_*`), per-request error capture, JSON persistence + reload, cancel support
- New `router/openai/batches_adapter.py`: `POST/GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`
- Tests: `tests/test_batch_manager.py` (12) â€” all pass

### Audit Log Query API (`/v1/audit/logs`)
- `AuditLog.query()` now supports action/actor/time-range/details filter + offset/limit; `recent_events` keeps "latest N" semantics
- New `router/audit_router.py`: unified query over `security` + `routing` JSONL sources, timestamp desc, `has_more`; `GET /v1/audit/sources`
- Tests: `tests/test_audit_log_query.py` (11) â€” all pass

### Realtime API (`/v1/realtime` WebSocket)
- New `runtime/realtime/realtime_session.py`: `RealtimeSession` state machine (session.update / conversation.item.create / ping / audio rejection) + `build_messages()` (instructions â†’ system, items â†’ user/assistant messages)
- New `router/realtime_router.py`: WS endpoint with Bearer/`?api_key=` auth, `session.created`, `response.create` â†’ `asyncio.to_thread(handle_chat)` â†’ `response.created` / `output_item.added` / `content_part.added` / chunked `text.delta` / `output_item.done` / `response.done` (failures â†’ `status: failed`)
- Tests: `tests/test_realtime_session.py` (12) â€” all pass

### Verification
- Related suites 70 passed (structured_output 20 + batch 12 + audit 11 + realtime 12 + orchestration 15); `router.openai_router` imports cleanly with all new routers wired

### Traces Dashboard Tab + OTEL Export (`/v1/traces`) (2026-08-11)
- `config/settings.py`: added `otel_endpoint` / `otel_export_enabled` / `traces_url` (dashboard traces source, defaults to 8001)
- New `router/traces_router.py`: `GET /v1/traces` (spans + trace_ids + execution-trace summaries), `GET /v1/traces/export?format=json|otlp`, `POST /v1/traces/export` (push OTLP to collector), `DELETE /v1/traces` (clear)
- `runtime/security/middleware.py`: `/v1/traces` + `/v1/traces/` prefix added to auth bypass (same localhost-observability pattern as `/api/metrics/`)
- `router/openai_router.py`: wired `traces_router` into the app
- `dashboard_server.py`: `GET /api/traces` proxy -> `{traces_url}/v1/traces`; failure returns empty arrays instead of crashing the dashboard
- `dashboard/templates/index.html`: new Traces tab with `#traces-panel`
- `dashboard/static/dashboard.js`: `renderTraces()` (spans table with name/trace/span/parent/elapsed_ms/attributes + execution traces table), refreshed in `refresh()`
- Verification: `GET /v1/traces` 200, `GET /v1/traces/export?format=json` 200, import smoke OK; `test_security.py` failures remain the 2 pre-existing `AIIH_API_KEY` env cases

### Traces Router Tests (2026-08-11)
- New `tests/test_traces_router.py`: 10 tests (empty/listed/filtered spans, execution summaries, json+otlp export shape w/ padded traceId=32/spanId=16, invalid format 400, POST export endpoint guard + dispatch, DELETE clear) -- all pass

### Tracer Bridge Fix + Request Trace Wiring (2026-08-11)
- `runtime/observability/tracing.py`: `TraceContext` is now a context manager (`__enter__/__exit__` -> `tracer.end_span`) with `set_attribute()` -- fixes `AttributeError` that would crash `execution_trace.start_trace()` (it used `with tracer.start_span(...) as span: span.set_attribute(...)` on a plain dataclass)
- `runtime/observability/execution_trace.py`: `start_trace` now opens the tracer `execution` span (with `execution_id`/`session_id` attributes) for the request's full lifetime; `end_trace` closes it via tracked `_tracer_ctx`; `clear()` also clears the tracer
- `router/openai_router.py`: new `request_trace_middleware` seeds `execution_trace_collector.start_trace` for `/v1/chat/completions` + `/v1/responses` (reads `x-session-id` header) and `end_trace` on an async `response.background` so streaming durations are captured; exceptions also close the trace
- Traces pipeline was previously dead code: `tracer` was never started in request paths and `execution_trace_collector.start_trace` was never called
- New `tests/test_execution_trace.py` (6): context-manager + set_attribute, start/end seeding + duration, noop end, clear resets bridge
- `tests/test_traces_router.py`: +1 integration test (chat request creates an execution trace even on auth reject)
- Verification: 54 passed across observability/orchestration/agent_loop/traces/execution_trace; E2E smoke: real `/v1/chat/completions` -> 1 execution trace + 1 `execution` span, duration captured

### Trace Memory Bounds (2026-08-11)
- `runtime/observability/tracing.py`: `Tracer(max_spans=1000)` + `_append_span` trims oldest spans when over cap
- `runtime/observability/execution_trace.py`: `ExecutionTraceCollector(max_traces=1000)` + `_evict_overflow` drops oldest trace (closing its tracer span) -- addresses Phase 33 audit item `tracer._spans + execution_trace._traces µL²M²z` now that every request seeds a trace
- `tests/test_execution_trace.py`: +2 (tracer span cap evicts oldest; collector trace cap evicts + closes evicted span)
- Verification: 35 passed (execution_trace/traces_router/observability)

### Per-User Token Usage in Dashboard Users Table (2026-08-11)
- `runtime/security/auth/token_tracker.py`: new `get_user_usage(db, user_ids)` -- per-user SQL aggregation (input/output/total + record_count), mirrors `get_api_key_usage`
- `dashboard_server.py` `GET /api/users`: now attaches `token_usage` per user (0 defaults)
- `dashboard/static/admin_users.js`: Users table gets a Tokens column (`toLocaleString`, hover tooltip with In/Out/requests) -- previously the Users table had no token usage at all
- `tests/test_token_tracker.py`: +1 (per-user aggregation)
- Note: `admin_api_keys.js` / `profile.js` Tokens columns + backend were already wired; if not visible in the browser, the running dashboard (9001) needs a restart to load the backend change (JS is served from disk, so it refreshes on reload)

### Token Usage Recording Verification (OpenAI path + Ollama/cloud providers) (2026-08-11)
- Correction: token usage recording was ALREADY fully wired -- all 4 `openai_handler` paths (`handle_chat`/`handle_streaming_chat`/`handle_responses`/`handle_streaming_responses`) call `_record_metrics` which, on success with `user_id != None`, calls `_record_token_usage` -> `record_token_usage` (DB). Provider is resolved generically, so Ollama/Gemini/NVIDIA NIM/Ollama Cloud/custom are all covered. Anthropic 8002 path records too.
- The guard `if user_id is None: return` is why usage stays flat: requests authenticated with the `AIIH_API_KEY` env key only attribute to a user when `AIIH_ADMIN_EMAIL` resolves (it does here -> admin user id 1). Registered dashboard API keys attribute per-key.
- `tests/test_orchestration.py`: +9 E2E tests -- `_record_token_usage` reached from chat (openai/ollama/gemini/nvidia_nim/ollama_cloud) + streaming (openai/ollama/ollama_cloud) with correct tokens/provider/model/user_id/api_key_id, plus the anonymous guard.
- Verification: 33 passed (orchestration + token_tracker)

### API Keys Tokens Column Not Visible (browser cache) (2026-08-11)
- Root cause: the Tokens column already exists in `admin_api_keys.js:19` and `profile.js:61` (committed in Phase 33), but the page templates loaded those JS files WITHOUT a cache-buster, so browsers served stale JS without the column.
- Fixed: added `?v={{ range(1,99999)|random }}` cache-buster to the JS includes in `admin_api_keys.html` / `profile.html` / `admin_users.html` (mirrors `index.html` pattern). Jinja render smoke-checked on all three.
- Action for user: hard refresh (Ctrl+F5) now picks up the new files; restart dashboard (9001) so `list_api_keys`/`list_my_api_keys` return `token_usage` (otherwise the column shows 0).

### API Keys vs Users token count mismatch resolved (2026-08-11)
- Question: Users shows 398,772 tokens but API Keys sum to 182,167 -- why?
- Root cause: get_user_usage counts ALL of a user's traffic (including rows with pi_key_id IS NULL), while get_api_key_usage only counts traffic authenticated with a registered API key. The gap (216,605 tokens / 28 requests) is exactly the pi_key_id IS NULL rows = env-key (AIIH_API_KEY) and dashboard-login traffic, which get user_id but no pi_key_id.
- untime/security/auth/token_tracker.py: new get_unkeyed_usage(db, user_ids=None) -- SQL aggregation over pi_key_id IS NULL.
- dashboard_server.py: list_api_keys (admin) and list_my_api_keys (self) now return {keys, unkeyed_usage}.
- dmin_api_keys.js / profile.js: summary line under the table -- Keyed / Unkeyed (env key/login) / Total pills (tooltip shows In/Out/requests); also rendered when key list is empty but unkeyed usage exists.
- Verified: unkeyed 216,605 + keyed 182,167 = 398,772 (matches Users exactly); 
ode --check on both JS files; 	est_token_tracker.py 8 passed.
- Action: restart dashboard (9001) for the new response shape; Ctrl+F5 for the new JS.

### agnes tool_choice 400 ­×´_¡]2026-08-11¡^
- Root cause¡G_anthropic_tool_choice_to_openai §â Anthropic {"type":"any"|"none"|"auto"} Âà¦¨ dict {"type":"required"} µ¥¡A¦ý OpenAI ToolChoice untagged enum ¥u±µ¨ü¦r¦ê "auto"/"none"/"required" ©Î {"type":"function","function":{"name"}}¡CÄY®æ Rust serde¡]agnes¡^¡÷ data did not match any variant of untagged enum ToolChoice¡C¥B /v1/messages ªº openai_payload ª½±µ°e adapter¡AµL¥¿³W¤Æ¨¾½u¡C
- Fix 1¡]nthropic_converter.py:_anthropic_tool_choice_to_openai¡^¡Gany¡÷"required"¡Bnone¡÷"none"¡Bauto¡÷"auto"¡Btool+name¡÷function object¡Btool µL name¡÷"auto"¡]¬Ò¬°¦Xªk variant¡^
- Fix 2¡]openai_handler.py¡^¡G·s¼W _normalize_tool_choice(tool_choice, tools) ¡X ¦r¦ê¥u¯d auto/none/required¡Fdict «¬ºAÂà¦¨¹ïÀ³¦r¦ê¡F{"type":"function"} ¯Ê function/name ©Î name ¤£¦b tools ¡÷ drop¡Fflat 
ame Äæ¦ì¥¿³W¤Æ¦¨ function object¡C®M¥Î©ó _normalize_payload_for_provider¡]ÂÐ»\ chat/streaming/responses + custom provider¡^»P handle_responses openai passthrough ¤À¤ä
- 	ests/test_tool_choice.py ¡X ·s¼W 21 tests¡]anthropic Âà´« 6 + normalize 11 + payload ¾ã¦X 4¡^
- ÅçÃÒ¡Gtest_tool_choice + test_prompt_caching + test_responses_e2e 39 passed¡Ftest_orchestration + test_server_tool_policy + test_web_server_tools 35 passed
- ?? »Ý­«±Ò 8002¡]anthropic_router¡^¸ü¤J¡F8001 ¥ç»Ý­«±Ò¡]²{¬° timeout¡^¡C¥t Ollama ¥Dµ{¦¡»Ý¥ý«ì´_¡A§_«h Ollama ¸ô¥Ñ¤´ 503
