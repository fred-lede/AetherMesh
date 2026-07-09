# Custom Cloud Providers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Dashboard CRUD UI + backend support for OpenAI-compatible custom cloud providers, eliminating hardcoded provider lists.

**Architecture:** Custom providers stored in `config/custom_providers.json` (name → api_type, base_url, api_key). Provider router checks this cache before falling back to built-in providers. Routing engine registers custom provider names dynamically via in-place list mutation. Dashboard API provides CRUD + probe endpoints.

**Tech Stack:** Python 3.14, FastAPI, JSON config, OpenAIAdapter (reused)

## Global Constraints

- Custom providers are OpenAI-compatible only (`api_type: "openai"`)
- Models stay in `config/models.yaml` with `provider: <name>` referencing the custom provider
- `custom_providers.json` is gitignored (contains API keys)
- No hot-reload without explicit reload call (`reload_custom_providers()`)
- Follow existing patterns in `settings.py`, `provider_router.py`, `routing_engine.py`, `dashboard_server.py`, `dashboard.js`

---

### Task 1: Config foundation — `.gitignore` + Settings helpers + schema

**Files:**
- Modify: `.gitignore`
- Modify: `config/settings.py`
- Create: `config/custom_providers.json` (empty JSON object `{}`)

**Interfaces:**
- Consumes: `Settings.config_path()` (already exists)
- Produces: `Settings.load_custom_providers() -> dict[str, Any]`, `Settings.save_custom_providers(data: dict[str, Any]) -> None`

- [ ] **Step 1.1: Add to `.gitignore`**

Append to `.gitignore`:
```
# Custom provider config (contains API keys)
config/custom_providers.json
```

- [ ] **Step 1.2: Create empty config**

```bash
echo '{}' > config/custom_providers.json
```

- [ ] **Step 1.3: Add `load_custom_providers()` to Settings**

After `load_yaml()` in `config/settings.py` (around line 174):

```python
def load_custom_providers(self) -> dict[str, Any]:
    path = self.config_path("custom_providers.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data
```

- [ ] **Step 1.4: Add `save_custom_providers()` to Settings**

After `save_cloud_credentials()` (around line 384):

```python
def save_custom_providers(self, data: dict[str, Any]) -> None:
    path = self.config_path("custom_providers.json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
```

---

### Task 2: Provider router integration

**Files:**
- Modify: `runtime/orchestration/provider_router.py`

**Interfaces:**
- Consumes: `settings.load_custom_providers()`, `OpenAIAdapter(api_key, base_url)`
- Produces: `_CUSTOM_PROVIDERS: dict`, `reload_custom_providers()`, `custom_provider_status()`, modified `adapter()`

- [ ] **Step 2.1: Add `_CUSTOM_PROVIDERS` module-level cache and loader**

In `provider_router.py`, after `_credential_pools` (line 44):

```python
_CUSTOM_PROVIDERS: dict[str, dict[str, Any]] = {}
_OPENAI_TYPES = {"openai"}
```

Add after `reload_credential_pools()` (around line 87):

```python
def _load_custom_providers() -> dict[str, dict[str, Any]]:
    data = settings.load_custom_providers()
    result: dict[str, dict[str, Any]] = {}
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        if not name or not isinstance(name, str):
            continue
        api_type = str(cfg.get("api_type", "openai")).lower()
        api_key = str(cfg.get("api_key", "")).strip()
        base_url = str(cfg.get("base_url", "")).strip()
        if api_key and base_url and api_type in _OPENAI_TYPES:
            result[name] = {"api_type": api_type, "base_url": base_url, "api_key": api_key}
    return result


def reload_custom_providers() -> dict[str, dict[str, Any]]:
    from runtime.orchestration.routing_engine import register_custom_providers, unregister_custom_providers
    old_names = set(_CUSTOM_PROVIDERS.keys())
    new_data = _load_custom_providers()
    _CUSTOM_PROVIDERS.clear()
    _CUSTOM_PROVIDERS.update(new_data)
    new_names = set(new_data.keys())
    to_remove = old_names - new_names
    if to_remove:
        unregister_custom_providers(list(to_remove))
    to_add = new_names - old_names
    if to_add:
        register_custom_providers(list(to_add), {n: new_data[n]["base_url"] for n in to_add})
    return dict(_CUSTOM_PROVIDERS)


def custom_provider_status() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, cfg in _CUSTOM_PROVIDERS.items():
        result.append({
            "name": name,
            "api_type": cfg["api_type"],
            "base_url": cfg["base_url"],
            "configured": bool(cfg.get("api_key")),
        })
    return result
```

- [ ] **Step 2.2: Load custom providers at import time**

After `_CUSTOM_PROVIDERS = {}` (new line from step 2.1):

```python
_CUSTOM_PROVIDERS.update(_load_custom_providers())
```

- [ ] **Step 2.3: Modify `adapter()` to check custom providers**

Before the final `raise ValueError` (around line 119):

```python
    if provider in _CUSTOM_PROVIDERS:
        cfg = _CUSTOM_PROVIDERS[provider]
        return OpenAIAdapter(api_key=cfg["api_key"], base_url=cfg["base_url"])
```



---

### Task 3: Routing engine dynamic registration

**Files:**
- Modify: `runtime/orchestration/routing_engine.py`

**Interfaces:**
- Consumes: custom provider names from `reload_custom_providers()`
- Produces: `register_custom_providers(names, name_base_urls)`, `unregister_custom_providers(names)` for provider_router to call

- [ ] **Step 3.1: Add registration/unregistration functions**

At end of `routing_engine.py` (before or after the module-level constants at lines 30-37):

```python
def register_custom_providers(names: list[str], name_base_urls: dict[str, str]) -> None:
    for name in names:
        if name not in CLOUD_PROVIDERS:
            CLOUD_PROVIDERS.append(name)
        if name not in ROUTING_PROVIDERS:
            ROUTING_PROVIDERS.append(name)
        if name not in CLOUD_PROVIDER_ENDPOINTS:
            base_url = name_base_urls.get(name, "")
            CLOUD_PROVIDER_ENDPOINTS[name] = ("", "", base_url)
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            if name not in cap_scores:
                cap_scores[name] = cap_scores.get("openai", 85)


def unregister_custom_providers(names: list[str]) -> None:
    for name in names:
        CLOUD_PROVIDERS[:] = [p for p in CLOUD_PROVIDERS if p != name]
        ROUTING_PROVIDERS[:] = [p for p in ROUTING_PROVIDERS if p != name]
        CLOUD_PROVIDER_ENDPOINTS.pop(name, None)
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            cap_scores.pop(name, None)
```

- [ ] **Step 3.2: Fix `_check_provider_credentials()` for custom providers**

Replace the static method (lines 136-142) with:

```python
@staticmethod
def _check_provider_credentials() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for provider, (_, api_key_env, _) in CLOUD_PROVIDER_ENDPOINTS.items():
        result[provider] = bool(os.getenv(api_key_env, "").strip())
    for provider in ROUTING_PROVIDERS:
        result.setdefault(provider, True)
    custom_path = settings.config_path("custom_providers.json")
    if custom_path.exists():
        try:
            custom_data = json.loads(custom_path.read_text(encoding="utf-8"))
            if isinstance(custom_data, dict):
                for name, cfg in custom_data.items():
                    if isinstance(cfg, dict) and str(cfg.get("api_key", "")).strip():
                        result[name] = True
        except (OSError, json.JSONDecodeError):
            pass
    return result
```

- [ ] **Step 3.3: Fix `_cloud_adapter_worker()` for custom providers**

Replace the method (lines 672-678) with:

```python
def _cloud_adapter_worker(self, provider: str) -> dict[str, Any]:
    if provider in CLOUD_PROVIDER_ENDPOINTS:
        base_url_env, api_key_env, default_base_url = CLOUD_PROVIDER_ENDPOINTS[provider]
        base_url = os.getenv(base_url_env, default_base_url).rstrip("/")
        if api_key_env:
            configured = bool(os.getenv(api_key_env, "").strip())
        else:
            configured = False
            custom_path = settings.config_path("custom_providers.json")
            if custom_path.exists():
                try:
                    custom_data = json.loads(custom_path.read_text(encoding="utf-8"))
                    if isinstance(custom_data, dict) and provider in custom_data:
                        cfg = custom_data[provider]
                        if isinstance(cfg, dict) and str(cfg.get("api_key", "")).strip():
                            configured = True
                except (OSError, json.JSONDecodeError):
                    pass
        return {
            "kind": "cloud_adapter",
            "base_url": base_url,
            "credential_configured": configured,
        }
    return {"kind": "cloud_adapter", "base_url": "", "credential_configured": False}
```

---

### Task 4: Dashboard API endpoints

**Files:**
- Modify: `dashboard/dashboard_server.py`

**Interfaces:**
- Consumes: `settings.load_custom_providers()`, `settings.save_custom_providers()`, `reload_custom_providers()`, `custom_provider_status()`, `_check_cloud_provider()`
- Produces: GET/POST/PUT/DELETE at `/api/custom-providers`, POST `/api/custom-providers/{name}/probe`, POST `/api/custom-providers/reload`

- [ ] **Step 4.1: Add imports**

Add to imports in `dashboard_server.py` (after line 26):
```python
from runtime.orchestration.provider_router import (
    credential_pool_status, reload_credential_pools,
    reload_custom_providers, custom_provider_status,
)
```

Replace the existing import on line 26:
```python
from runtime.orchestration.provider_router import credential_pool_status, reload_credential_pools
```
with:
```python
from runtime.orchestration.provider_router import (
    credential_pool_status, reload_credential_pools,
    custom_provider_status, reload_custom_providers,
)
```

- [ ] **Step 4.2: Add custom provider CRUD endpoints**

After the credentials endpoints (after line 985), add:

```python
@api.get("/custom-providers")
def custom_providers_list() -> dict[str, Any]:
    data = settings.load_custom_providers()
    safe = {}
    for name, cfg in data.items():
        if isinstance(cfg, dict):
            safe[name] = {
                "name": name,
                "api_type": cfg.get("api_type", "openai"),
                "base_url": cfg.get("base_url", ""),
                "has_key": bool(str(cfg.get("api_key", "")).strip()),
            }
    return {"providers": safe}


@api.post("/custom-providers")
def custom_providers_create(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Provider name is required")
    base_url = str(body.get("base_url", "")).strip().rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="Base URL is required")
    api_key = str(body.get("api_key", "")).strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    data = settings.load_custom_providers()
    if name in data:
        raise HTTPException(status_code=409, detail=f"Provider '{name}' already exists")
    data[name] = {"api_type": "openai", "base_url": f"{base_url}/v1" if "/v1" not in base_url else base_url, "api_key": api_key}
    settings.save_custom_providers(data)
    reload_custom_providers()
    return {"ok": True, "name": name}


@api.put("/custom-providers/{name}")
def custom_providers_update(name: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    data = settings.load_custom_providers()
    if name not in data:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    cfg = data[name]
    if not isinstance(cfg, dict):
        cfg = {}
    if "base_url" in body:
        base_url = str(body["base_url"]).strip().rstrip("/")
        if base_url:
            cfg["base_url"] = f"{base_url}/v1" if "/v1" not in base_url else base_url
    if "api_key" in body:
        val = str(body["api_key"]).strip()
        if val:
            cfg["api_key"] = val
    data[name] = cfg
    settings.save_custom_providers(data)
    reload_custom_providers()
    return {"ok": True, "name": name}


@api.delete("/custom-providers/{name}")
def custom_providers_delete(name: str) -> dict[str, Any]:
    data = settings.load_custom_providers()
    if name not in data:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    del data[name]
    settings.save_custom_providers(data)
    reload_custom_providers()
    return {"ok": True, "name": name}
```

- [ ] **Step 4.3: Add probe endpoint for custom providers**

After the `custom_providers_delete` endpoint:

```python
@api.post("/custom-providers/{name}/probe")
def custom_providers_probe(name: str) -> dict[str, Any]:
    data = settings.load_custom_providers()
    cfg = data.get(name)
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    base_url = str(cfg.get("base_url", "")).rstrip("/")
    api_key = str(cfg.get("api_key", "")).strip()
    if not api_key:
        return {"name": name, "ok": False, "status": "not_configured", "message": "No API key"}
    headers = {"Authorization": f"Bearer {api_key}"}
    endpoints_to_try = ["/models", "/api/tags"]
    for endpoint in endpoints_to_try:
        try:
            response = get_session().get(f"{base_url}{endpoint}", headers=headers, timeout=2)
            if response.ok:
                data_json = response.json()
                model_count = 0
                if isinstance(data_json, dict):
                    if "data" in data_json:
                        model_count = len(data_json["data"])
                    elif "models" in data_json:
                        model_count = len(data_json["models"])
                elif isinstance(data_json, list):
                    model_count = len(data_json)
                return {
                    "name": name,
                    "ok": True,
                    "status": "healthy",
                    "base_url": base_url,
                    "model_count": model_count,
                    "latency_ms": int(response.elapsed.total_seconds() * 1000),
                }
        except requests.RequestException:
            continue
    return {
        "name": name,
        "ok": False,
        "status": "unreachable",
        "base_url": base_url,
        "message": f"Failed to connect to {base_url}",
    }
```

- [ ] **Step 4.4: Add reload endpoint**

```python
@api.post("/custom-providers/reload")
def custom_providers_reload() -> dict[str, Any]:
    reloaded = reload_custom_providers()
    return {"ok": True, "providers": list(reloaded.keys())}
```

- [ ] **Step 4.5: Include custom providers in `_build_overview()`**

In `_build_overview()` (around line 867, after `cloud_providers = _check_cloud_providers()`):

```python
    custom_providers = custom_provider_status()
```

And add to the return dict (around line 887):

```python
        "custom_providers": custom_providers,
```

---

### Task 5: Dashboard UI panel

**Files:**
- Modify: `dashboard/static/dashboard.js`
- Modify: `dashboard/templates/index.html`

**Interfaces:**
- Consumes: `data.custom_providers` from SSE/overview API, CRUD endpoints from Task 4
- Produces: Rendered "Custom Providers" section in Providers tab

- [ ] **Step 5.1: Add HTML section to the Providers tab**

In `dashboard/templates/index.html`, find the Providers tab content (`<div class="tab-content" data-tab="providers">`) and add inside it, after the credential section:

```html
<section class="section card">
  <div class="section-head">
    <h2>Custom Providers</h2>
    <span class="mono" style="color: var(--muted);">OpenAI-compatible providers added via Dashboard</span>
  </div>
  <div id="custom-provider-section">
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
      <input id="cp-name" class="control-input" type="text" placeholder="Provider name (e.g. agnes)" style="flex:1;min-width:140px;">
      <input id="cp-baseurl" class="control-input" type="text" placeholder="Base URL (e.g. https://api.agnes.ai/v1)" style="flex:2;min-width:200px;">
      <input id="cp-apikey" class="control-input" type="password" placeholder="API key" style="flex:1;min-width:140px;">
      <button class="btn btn-primary" onclick="addCustomProvider(event)">Add</button>
    </div>
    <div id="custom-provider-grid" class="provider-grid"></div>
  </div>
</section>
```

- [ ] **Step 5.2: Add JS render function for custom providers**

In `dashboard/static/dashboard.js`, add a new function:

```javascript
function renderCustomProviders(data) {
  const grid = document.getElementById('custom-provider-grid');
  if (!grid) return;
  const providers = data.custom_providers || [];
  if (!providers.length) {
    grid.innerHTML = '<div style="color:var(--muted);padding:12px;">No custom providers configured.</div>';
    return;
  }
  grid.innerHTML = providers.map(p => `
    <div class="provider-card" id="cp-card-${escapeJsString(p.name)}">
      <div class="provider-top">
        <div>
          <div class="provider-name">${escapeHtml(p.name)}</div>
          <div class="provider-sub">${escapeHtml(p.base_url)}</div>
        </div>
        <span class="pill ${statusClass(p.configured ? 'ok' : 'bad')}">${p.configured ? 'configured' : 'no key'}</span>
      </div>
      <div class="provider-actions" style="display:flex;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid var(--line);">
        <button class="btn btn-primary" onclick="probeCustomProvider('${escapeJsString(p.name)}')">Test</button>
        <button class="btn" onclick="editCustomProvider('${escapeJsString(p.name)}')">Edit</button>
        <button class="btn btn-danger" onclick="deleteCustomProvider('${escapeJsString(p.name)}')">Delete</button>
        <span id="cp-probe-${escapeJsString(p.name)}" style="font-size:0.8rem;color:var(--muted);align-self:center;"></span>
      </div>
    </div>
  `).join('');
}
```

- [ ] **Step 5.3: Wire into `renderOverview()`**

In the `renderOverview(data)` function, add after the credential rendering call:

```javascript
  renderCustomProviders(data);
```

- [ ] **Step 5.4: Add CRUD action functions**

In `dashboard/static/dashboard.js`, add:

```javascript
async function addCustomProvider(event) {
  const btn = event.currentTarget;
  const restore = setButtonBusy(btn, 'Adding...');
  try {
    const name = document.getElementById('cp-name').value.trim();
    const baseUrl = document.getElementById('cp-baseurl').value.trim();
    const apiKey = document.getElementById('cp-apikey').value.trim();
    if (!name || !baseUrl || !apiKey) {
      setOperationStatus('All fields are required.', 'bad');
      return;
    }
    await mutateDashboard('/api/custom-providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, base_url: baseUrl, api_key: apiKey }),
    });
    document.getElementById('cp-name').value = '';
    document.getElementById('cp-baseurl').value = '';
    document.getElementById('cp-apikey').value = '';
    await refresh();
    setOperationStatus(`Provider "${name}" added.`, 'ok');
  } catch (err) {
    setOperationStatus(`Failed: ${err.message}`, 'bad');
  } finally {
    restore();
  }
}

async function probeCustomProvider(name) {
  const span = document.getElementById(`cp-probe-${name}`);
  if (span) span.textContent = 'Probing...';
  try {
    const result = await mutateDashboard(`/api/custom-providers/${encodeURIComponent(name)}/probe`, { method: 'POST' });
    if (span) span.textContent = result.ok
      ? `OK (${result.latency_ms || '?'}ms, ${result.model_count || 0} models)`
      : `Failed: ${result.status}`;
  } catch (err) {
    if (span) span.textContent = `Error: ${err.message}`;
  }
}

async function deleteCustomProvider(name) {
  if (!confirm(`Delete provider "${name}"?`)) return;
  try {
    await mutateDashboard(`/api/custom-providers/${encodeURIComponent(name)}`, { method: 'DELETE' });
    await refresh();
    setOperationStatus(`Provider "${name}" deleted.`, 'ok');
  } catch (err) {
    setOperationStatus(`Failed: ${err.message}`, 'bad');
  }
}

function editCustomProvider(name) {
  const nameInput = document.getElementById('cp-name');
  const baseUrlInput = document.getElementById('cp-baseurl');
  const apiKeyInput = document.getElementById('cp-apikey');
  const addBtn = document.querySelector('#custom-provider-section .btn-primary');
  nameInput.value = name;
  nameInput.readOnly = true;
  addBtn.textContent = 'Save';
  addBtn.onclick = async function saveEdit(event) {
    const restore = setButtonBusy(addBtn, 'Saving...');
    try {
      const baseUrl = baseUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      const body = {};
      if (baseUrl) body.base_url = baseUrl;
      if (apiKey) body.api_key = apiKey;
      await mutateDashboard(`/api/custom-providers/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      nameInput.readOnly = false;
      nameInput.value = '';
      baseUrlInput.value = '';
      apiKeyInput.value = '';
      addBtn.textContent = 'Add';
      addBtn.onclick = addCustomProvider;
      await refresh();
      setOperationStatus(`Provider "${name}" updated.`, 'ok');
    } catch (err) {
      setOperationStatus(`Failed: ${err.message}`, 'bad');
    } finally {
      restore();
    }
  };
}
```

---

### Task 6: Tests

**Files:**
- Create: `tests/test_custom_providers.py`

**Test coverage:**
- `settings.load_custom_providers()` and `save_custom_providers()` with temp file
- `_load_custom_providers()` in provider_router — filters valid entries, skips invalid
- `reload_custom_providers()` — reads file, updates cache
- `adapter()` — returns OpenAIAdapter for custom provider name
- `register_custom_providers()` / `unregister_custom_providers()` in routing_engine
- Dashboard custom provider CRUD endpoints via TestClient
- Probe endpoint

- [ ] **Step 6.1: Write and run tests**

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from config.settings import Settings
from runtime.orchestration.provider_router import (
    OpenAIAdapter, _CUSTOM_PROVIDERS, _load_custom_providers,
    reload_custom_providers, adapter,
)
from runtime.orchestration.routing_engine import (
    CLOUD_PROVIDERS, ROUTING_PROVIDERS, CLOUD_PROVIDER_ENDPOINTS,
    CAPABILITY_PROVIDER_SCORES,
    register_custom_providers, unregister_custom_providers,
)


# ── helpers ──────────────────────────────────────────────────────────

def _with_temp_custom_providers(data: dict) -> Path:
    """Write *data* to a temp JSON and patch settings.config_path to return it."""
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(data), encoding="utf-8")
    return tmp


# ── Settings ─────────────────────────────────────────────────────────

class TestSettingsLoadCustomProviders:
    def test_returns_empty_when_file_missing(self):
        s = Settings()
        with patch.object(s, "config_path", return_value=Path("/nonexistent/file.json")):
            assert s.load_custom_providers() == {}

    def test_returns_empty_on_invalid_json(self):
        s = Settings()
        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_text("not json", encoding="utf-8")
        with patch.object(s, "config_path", return_value=tmp):
            assert s.load_custom_providers() == {}

    def test_returns_parsed_data(self):
        s = Settings()
        data = {"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}
        tmp = _with_temp_custom_providers(data)
        with patch.object(s, "config_path", return_value=tmp):
            assert s.load_custom_providers() == data


class TestSettingsSaveCustomProviders:
    def test_writes_valid_json(self):
        s = Settings()
        tmp = Path(tempfile.mktemp(suffix=".json"))
        with patch.object(s, "config_path", return_value=tmp):
            data = {"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}
            s.save_custom_providers(data)
            assert json.loads(tmp.read_text(encoding="utf-8")) == data


# ── Provider Router ──────────────────────────────────────────────────

class TestLoadCustomProviders:
    def test_filters_valid_entries(self):
        data = {
            "agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"},
            "bad": {"api_type": "openai", "base_url": "", "api_key": ""},
            "not_dict": "string",
        }
        tmp = _with_temp_custom_providers(data)
        s = Settings()
        with patch.object(s, "config_path", return_value=tmp):
            with patch("runtime.orchestration.provider_router.settings", s):
                result = _load_custom_providers()
                assert "agnes" in result
                assert result["agnes"]["base_url"] == "https://api.agnes.ai/v1"
                assert "bad" not in result
                assert "not_dict" not in result


class TestAdapterCustomProviders:
    def test_returns_openai_adapter_for_custom_provider(self):
        with patch.dict(_CUSTOM_PROVIDERS, {
            "agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-agnes"},
        }, clear=True):
            result = adapter("agnes")
            assert isinstance(result, OpenAIAdapter)
            assert result.base_url == "https://api.agnes.ai/v1"
            assert result.api_key == "sk-agnes"

    def test_raises_for_unknown_custom_provider(self):
        with patch.dict(_CUSTOM_PROVIDERS, {}, clear=True):
            with pytest.raises(ValueError, match="Unsupported provider: unknown"):
                adapter("unknown")

    def test_raises_for_custom_provider_without_api_key(self):
        with patch.dict(_CUSTOM_PROVIDERS, {
            "nokey": {"api_type": "openai", "base_url": "https://example.com/v1", "api_key": ""},
        }, clear=True):
            with pytest.raises(Exception):  # ProviderError from OpenAIAdapter
                adapter("nokey")


class TestReloadCustomProviders:
    def test_reloads_and_updates_cache(self):
        data = {"test_prov": {"api_type": "openai", "base_url": "https://test.ai/v1", "api_key": "sk-test"}}
        tmp = _with_temp_custom_providers(data)
        s = Settings()
        with patch.object(s, "config_path", return_value=tmp):
            with patch("runtime.orchestration.provider_router.settings", s):
                with patch("runtime.orchestration.provider_router.register_custom_providers") as mock_reg:
                    with patch("runtime.orchestration.provider_router.unregister_custom_providers"):
                        _CUSTOM_PROVIDERS.clear()
                        result = reload_custom_providers()
                        assert "test_prov" in result
                        mock_reg.assert_called_once_with(["test_prov"], {"test_prov": "https://test.ai/v1"})


# ── Routing Engine ───────────────────────────────────────────────────

class TestRegisterCustomProviders:
    def setup_method(self):
        # Save originals
        self._orig_cloud = list(CLOUD_PROVIDERS)
        self._orig_routing = list(ROUTING_PROVIDERS)
        self._orig_endpoints = dict(CLOUD_PROVIDER_ENDPOINTS)
        self._orig_scores = {k: dict(v) for k, v in CAPABILITY_PROVIDER_SCORES.items()}

    def teardown_method(self):
        CLOUD_PROVIDERS[:] = self._orig_cloud
        ROUTING_PROVIDERS[:] = self._orig_routing
        CLOUD_PROVIDER_ENDPOINTS.clear()
        CLOUD_PROVIDER_ENDPOINTS.update(self._orig_endpoints)
        CAPABILITY_PROVIDER_SCORES.clear()
        CAPABILITY_PROVIDER_SCORES.update(self._orig_scores)

    def test_registers_provider_in_all_lists(self):
        register_custom_providers(["testai"], {"testai": "https://test.ai/v1"})
        assert "testai" in CLOUD_PROVIDERS
        assert "testai" in ROUTING_PROVIDERS
        assert CLOUD_PROVIDER_ENDPOINTS["testai"] == ("", "", "https://test.ai/v1")
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            assert "testai" in cap_scores
            assert cap_scores["testai"] == cap_scores.get("openai", 85)

    def test_unregisters_provider(self):
        register_custom_providers(["testai"], {"testai": "https://test.ai/v1"})
        unregister_custom_providers(["testai"])
        assert "testai" not in CLOUD_PROVIDERS
        assert "testai" not in ROUTING_PROVIDERS
        assert "testai" not in CLOUD_PROVIDER_ENDPOINTS
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            assert "testai" not in cap_scores


# ── Dashboard API ────────────────────────────────────────────────────

class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from dashboard.dashboard_server import app
        return TestClient(app)

    def test_custom_providers_list_empty(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers", return_value={}):
            resp = client.get("/api/custom-providers")
            assert resp.status_code == 200
            data = resp.json()
            assert data["providers"] == {}

    def test_custom_providers_create(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers", return_value={}):
            with patch("dashboard.dashboard_server.settings.save_custom_providers") as mock_save:
                with patch("dashboard.dashboard_server.reload_custom_providers") as mock_reload:
                    resp = client.post("/api/custom-providers", json={
                        "name": "agnes", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"
                    })
                    assert resp.status_code == 200
                    assert resp.json()["name"] == "agnes"
                    mock_save.assert_called_once()
                    mock_reload.assert_called_once()

    def test_custom_providers_create_duplicate(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers",
                   return_value={"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}):
            resp = client.post("/api/custom-providers", json={
                "name": "agnes", "base_url": "https://api.agnes.ai/v2", "api_key": "sk-other"
            })
            assert resp.status_code == 409

    def test_custom_providers_delete(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers",
                   return_value={"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}):
            with patch("dashboard.dashboard_server.settings.save_custom_providers"):
                with patch("dashboard.dashboard_server.reload_custom_providers"):
                    resp = client.delete("/api/custom-providers/agnes")
                    assert resp.status_code == 200

    def test_custom_providers_delete_not_found(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers", return_value={}):
            resp = client.delete("/api/custom-providers/nonexistent")
            assert resp.status_code == 404

    def test_custom_providers_probe(self, client):
        with patch("dashboard.dashboard_server.settings.load_custom_providers",
                   return_value={"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}):
            with patch("dashboard.dashboard_server.get_session") as mock_session:
                mock_resp = MagicMock()
                mock_resp.ok = True
                mock_resp.json.return_value = {"data": [{"id": "model-1"}]}
                mock_resp.elapsed.total_seconds.return_value = 0.15
                mock_session.return_value.get.return_value = mock_resp
                resp = client.post("/api/custom-providers/agnes/probe")
                assert resp.status_code == 200
                data = resp.json()
                assert data["ok"] is True
                assert data["model_count"] == 1
                assert data["latency_ms"] == 150
```
