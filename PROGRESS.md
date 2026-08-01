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
