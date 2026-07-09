# Custom Cloud Providers — Design Doc

## Problem

Adding a new OpenAI-compatible cloud provider (e.g., Agnes, Together, Groq) requires
editing 7+ files with hardcoded lists (`provider_router.py`, `routing_engine.py`,
`credential_pool.py`, dashboard server/JS, etc.). There is no UI to manage
provider connections.

## Goal

Dashboard UI to add/edit/delete OpenAI-compatible cloud providers, specifying
only base URL + API key. Models are still defined in `config/models.yaml` with
`provider: <name>` referencing the custom provider.

## Non-Goals

- Anthropic or other non-OpenAI-compatible provider types (future)
- Dynamic model registration (models stay in `models.yaml`)
- Hot-reload without restart (runtime reload supported via API call, but requires
  re-registration in routing engine)

## Storage

New file `config/custom_providers.json`:

```json
{
  "agnes": {
    "api_type": "openai",
    "base_url": "https://api.agnes.ai/v1",
    "api_key": "sk-xxx"
  }
}
```

- **Top-level key** = provider name (e.g., `agnes`), must match `provider:` in `models.yaml`
- **`api_type`**: only `"openai"` for now (extensible for future types)
- **`base_url`**: full URL including `/v1` path
- **`api_key`**: authentication key

Gitignore: `custom_providers.json` is not gitignored (unlike `credentials.json`),
because it's a shared config that defines the provider, not just secrets.
Actually — it contains API keys, so it SHOULD be gitignored. Add to `.gitignore`.

## Routing Integration

### `runtime/orchestration/provider_router.py`

- New `_CUSTOM_PROVIDERS: dict[str, dict]` cache, populated from `custom_providers.json`
  at import time
- `adapter(provider)` method: when provider not in `_CLOUD_ADAPTERS`, check
  `_CUSTOM_PROVIDERS`. If found, create and cache an `OpenAIAdapter` instance
  configured with the custom `base_url` and `api_key`
- New function `reload_custom_providers()`: re-reads JSON, clears adapter cache,
  re-registers in routing engine
- New function `custom_provider_status()`: returns list of custom providers with
  health status for Dashboard

### `runtime/orchestration/routing_engine.py`

- `CLOUD_PROVIDERS` set becomes dynamic: custom provider names are added at the
  same point where routing state is initialized
- `CLOUD_PROVIDER_ENDPOINTS` entries for custom providers are generated from
  config at lookup time (not static dict)
- `ROUTING_PROVIDERS` similarly updated dynamically

### `openai_adapter.py` impact

Minimal — the adapter already accepts `base_url` and `api_key` as constructor
parameters (from env vars). For custom providers, these are passed explicitly
instead of read from env. Verify the constructor signature supports this.

## Dashboard API

All new endpoints under `/api/custom-providers`:

| Method | Path | Request | Response | Description |
|--------|------|---------|----------|-------------|
| GET | `/api/custom-providers` | — | `{providers: {name: {...}}}` | List all custom providers |
| POST | `/api/custom-providers` | `{name, base_url, api_key}` | `{status: "ok"}` | Add new provider |
| PUT | `/api/custom-providers/{name}` | `{base_url?, api_key?}` | `{status: "ok"}` | Edit provider |
| DELETE | `/api/custom-providers/{name}` | — | `{status: "ok"}` | Delete provider |
| POST | `/api/custom-providers/{name}/probe` | — | `{status: "ok"/"error", latency_ms}` | Test connection (GET /models or simple chat) |
| POST | `/api/custom-providers/reload` | — | `{status: "ok", providers: [...]}` | Reload config + re-register in routing |

On write (POST/PUT/DELETE): update `custom_providers.json` on disk, then
call `reload_custom_providers()`.

## Dashboard UI

New "Custom Providers" section in Dashboard, rendered alongside existing
"Cloud Credentials" or as its own panel.

### Provider Card (each custom provider)
- Name badge
- Status indicator (green/red/gray for healthy/unreachable/not probed)
- Base URL (truncated, copyable)
- Edit button → inline form
- Delete button → confirm dialog
- "Test Connection" button

### Add Provider Form
- Provider name (text input)
- Base URL (text input, placeholder `https://api.example.com/v1`)
- API key (password input)
- Save button

### Edit Provider Form
- Same fields, pre-filled (API key masked, optional change)

No pagination or search needed for expected scale (< 20 providers).

## Error Handling

- Duplicate provider name: 409 Conflict
- Non-existent provider on PUT/DELETE: 404 Not Found
- Invalid base_url on probe: 400 Bad Request + probe returns error status
- File write failure: 500 Internal Server Error
- Probe timeout: return error status, not 500

## Implementation Order

1. Add `config/custom_providers.json` to `.gitignore`
2. Create `_CUSTOM_PROVIDERS` cache and `reload_custom_providers()` in
   `provider_router.py`
3. Modify `adapter()` to instantiate `OpenAIAdapter` for custom providers
4. Dynamic registration in `routing_engine.py` (`CLOUD_PROVIDERS`, etc.)
5. Dashboard API endpoints in `dashboard/dashboard_server.py`
6. Dashboard UI panel in `dashboard/static/dashboard.js`
7. Probe endpoint (GET /v1/models or simple chat)
