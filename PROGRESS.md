# Progress

## 2026-07-09 — Post-deployment fixes: probe, enable-state, credential cards

### Fixes
- **Probe**: `_probe_provider()` raised 404 for custom providers — added `is_custom_provider()` check to delegate to `_probe_custom_provider()`
- **Route disabled on startup**: `register_custom_providers()` now calls `routing_engine.set_provider_enabled(name, True)`, and `_load_state()` no longer filters out unknown providers (preserves prior manual toggle across restarts)
- **Cloud Credentials cards**: template + JS now dynamically include custom providers; `renderCredentials()` injects/removes credential cards when providers are added/deleted without page refresh
- **Test state leakage**: module-level cleanup deletes stale `routing_state.yaml` before it's loaded; teardown restores `_provider_enabled`
