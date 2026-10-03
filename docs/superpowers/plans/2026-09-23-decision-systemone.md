# Decision Models (`/v1/systemone`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Ollama decision models through AetherMesh with a `POST /v1/systemone` endpoint that routes the request to the correct Ollama worker and returns its result.

**Architecture:** A new `router/decision_router.py` resolves the model's provider/worker via the existing `provider_router`, calls a new `OllamaAdapter.systemone()` (a thin `POST {worker}/v1/systemone` passthrough), and returns Ollama's JSON. A `decision` capability is added to the vocabulary and the Dashboard capability list; the three models are registered manually in `config/models.yaml`.

**Tech Stack:** Python 3.14, FastAPI, requests, pytest (`asyncio_mode = auto`).

**Spec:** `docs/superpowers/specs/2026-09-23-decision-systemone-design.md`

## Global Constraints

- Python: 4-space indent, `from __future__ import annotations` at top of new modules, type hints on public APIs, no docstrings/comments unless required.
- Endpoint: `POST /v1/systemone`, non-streaming, mounted on the OpenAI router (port 8001); auth via the existing security middleware.
- Sync adapter calls must run under `asyncio.to_thread`.
- Adapter HTTP failures raise `providers.base.ProviderError` (message = response text).
- Non-`ollama` provider → HTTP **501**; missing `model` → HTTP **400**; upstream failure → HTTP **502** with the upstream message in `detail`.
- Capability value is the exact string `decision`.
- Verified upstream contract (Ollama): request `{model, state: string|object|array, questions: {key: {type: "choice"|"noul"|"score", instructions: <nonempty string|object|array>, criteria}}}`, 1–64 questions; response `{model, answers, usage}`.
- Test command: `.venv\Scripts\python.exe -m pytest <path> -v` (add `--basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm` — the default pytest temp root is locked).

## Review Focus

- An empty or >64-entry `questions` map must return the upstream 400 message, not a 500 or a swallowed error.
- `state` given as an object or array must be forwarded byte-for-byte (the endpoint must not coerce it to a string).
- A model whose provider is not `ollama` must return 501 — never a silent call to a local worker.
- An upstream non-200 (e.g. `{"error":"model is required"}`) must preserve a clear status + message to the caller.
- Adapter connection failure (worker down) must surface as a 502 with a message, not an unhandled stack trace.

---

### Task 1: Add the `decision` capability to the vocabulary and Dashboard

**Files:**
- Modify: `providers/registry.py:11-50`
- Modify: `dashboard/dashboard_server.py` (`_CAPABILITY_GROUPS`, currently ~line 1145)
- Test: `tests/test_providers_registry.py` (append)
- Test: `tests/test_models_api.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `Capability.DECISION` (`"decision"`), `CAPABILITY_ALIASES["decision"]`, and a `_CAPABILITY_GROUPS` entry `{"value":"decision","label":"Decision","group":"other"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers_registry.py`:

```python
def test_decision_capability_is_canonical():
    from providers.registry import CAPABILITY_ALIASES, Capability, parse_capabilities

    assert Capability.DECISION.value == "decision"
    assert CAPABILITY_ALIASES["decision"] is Capability.DECISION
    assert parse_capabilities(["decision"]) == {Capability.DECISION}
```

Append to `tests/test_models_api.py`:

```python
def test_capabilities_endpoint_includes_decision(client: TestClient):
    values = {c["value"] for c in client.get("/api/models/capabilities").json()["capabilities"]}
    assert "decision" in values
```

Append to `tests/test_model_registry_store.py`:

```python
def test_validate_accepts_decision_capability():
    entry = {"name": "nimble:9b", "provider": "ollama",
             "worker_bindings": [{"node_id": "node-01", "port": 11434}], "capabilities": ["decision"]}
    _, errors = store.validate_model(entry)
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_providers_registry.py::test_decision_capability_is_canonical tests/test_models_api.py::test_capabilities_endpoint_includes_decision tests/test_model_registry_store.py::test_validate_accepts_decision_capability -v --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: FAIL — `AttributeError: DECISION` / assertion on missing `decision`.

- [ ] **Step 3: Write minimal implementation**

In `providers/registry.py`, add to the `Capability` enum (after `DOCUMENTS = "documents"`):

```python
    DECISION = "decision"
```

In `CAPABILITY_ALIASES` (after the `"documents"` entries), add:

```python
    "decision": Capability.DECISION,
```

In `dashboard/dashboard_server.py`, add to `_CAPABILITY_GROUPS` (after the `rerank` entry):

```python
    {"value": "decision", "label": "Decision", "group": "other"},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_providers_registry.py tests/test_models_api.py tests/test_model_registry_store.py -q --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/registry.py dashboard/dashboard_server.py tests/test_providers_registry.py tests/test_models_api.py tests/test_model_registry_store.py
git commit -m "feat: add decision capability to vocabulary and dashboard"
```

---

### Task 2: `OllamaAdapter.systemone()`

**Files:**
- Modify: `providers/ollama_adapter.py` (add method after `responses`, ~line 58)
- Test: `tests/test_ollama_systemone.py` (create)

**Interfaces:**
- Consumes: `providers.http_client.get_session`, `providers.base.ProviderError`, `config.settings.settings.request_timeout_s`.
- Produces: `OllamaAdapter.systemone(payload: dict[str, Any]) -> dict[str, Any]` — POSTs to `{self.base_url}/v1/systemone` and returns the parsed JSON; raises `ProviderError(response.text)` on non-2xx.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ollama_systemone.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from providers.base import ProviderError
from providers.ollama_adapter import OllamaAdapter


def test_systemone_posts_and_returns_json():
    adapter = OllamaAdapter("http://x:11434")
    resp = MagicMock(ok=True)
    resp.json.return_value = {"model": "nimble:9b", "answers": {"intent": {"type": "choice", "choice": "book"}}}
    with patch("providers.ollama_adapter.get_session") as factory:
        factory.return_value.post.return_value = resp
        out = adapter.systemone({"model": "nimble:9b", "state": "x", "questions": {"intent": {"type": "choice"}}})
    assert out["answers"]["intent"]["choice"] == "book"
    assert factory.return_value.post.call_args.args[0] == "http://x:11434/v1/systemone"


def test_systemone_raises_on_error():
    adapter = OllamaAdapter("http://x:11434")
    resp = MagicMock(ok=False)
    resp.text = "model is required"
    with patch("providers.ollama_adapter.get_session") as factory:
        factory.return_value.post.return_value = resp
        with pytest.raises(ProviderError):
            adapter.systemone({"model": "nimble:9b"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ollama_systemone.py -v --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: FAIL — `AttributeError: 'OllamaAdapter' object has no attribute 'systemone'`.

- [ ] **Step 3: Write minimal implementation**

In `providers/ollama_adapter.py`, add after the `responses` method:

```python
    def systemone(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = get_session().post(
            f"{self.base_url}/v1/systemone",
            json=payload,
            timeout=settings.request_timeout_s,
        )
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text)
        return response.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ollama_systemone.py -v --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add providers/ollama_adapter.py tests/test_ollama_systemone.py
git commit -m "feat: add OllamaAdapter.systemone for /v1/systemone"
```

---

### Task 3: `router/decision_router.py` and wiring

**Files:**
- Create: `router/decision_router.py`
- Modify: `router/openai_router.py` (imports ~line 39; include alongside other `app.include_router(...)` calls)
- Test: `tests/test_decision_router.py` (create)

**Interfaces:**
- Consumes: `providers.base.ProviderError`, `runtime.orchestration.model_registry_store.get_models`, `runtime.orchestration.provider_router.resolve_provider(model, registry)`, `provider_router.adapter(provider, worker)`, Task 2's `OllamaAdapter.systemone`.
- Produces: module-level `router` (`APIRouter`) exposing `POST /v1/systemone`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_decision_router.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from providers.base import ProviderError
from router import decision_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(decision_router.router)
    return TestClient(app)


def test_missing_model_returns_400():
    resp = _client().post("/v1/systemone", json={"state": "x", "questions": {}})
    assert resp.status_code == 400


def test_routes_to_ollama_adapter():
    adapter = MagicMock()
    adapter.systemone.return_value = {"model": "nimble:9b", "answers": {"intent": {"type": "choice", "choice": "book"}}}
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        resp = _client().post("/v1/systemone", json={"model": "nimble:9b", "state": "x", "questions": {"intent": {}}})
    assert resp.status_code == 200
    assert resp.json()["answers"]["intent"]["choice"] == "book"
    adapter.systemone.assert_called_once()


def test_unsupported_provider_returns_501():
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("agnes", None)):
        resp = _client().post("/v1/systemone", json={"model": "x", "state": "y", "questions": {}})
    assert resp.status_code == 501


def test_upstream_error_is_preserved():
    adapter = MagicMock()
    adapter.systemone.side_effect = ProviderError("questions must contain 1-64 fields")
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        resp = _client().post("/v1/systemone", json={"model": "m", "state": "s", "questions": {}})
    assert resp.status_code == 502
    assert "questions must contain" in resp.json()["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_decision_router.py -v --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.decision_router'`.

- [ ] **Step 3: Write minimal implementation**

Create `router/decision_router.py`:

```python
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from providers.base import ProviderError
from runtime.orchestration import model_registry_store, provider_router

logger = logging.getLogger("router.decision")
router = APIRouter(tags=["decision"])


@router.post("/v1/systemone")
async def systemone(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    registry = {"models": model_registry_store.get_models()}
    provider, worker = provider_router.resolve_provider(model, registry)
    if provider != "ollama":
        raise HTTPException(
            status_code=501,
            detail=f"provider '{provider}' does not support decision (/v1/systemone) yet",
        )
    adapter = provider_router.adapter(provider, worker)
    try:
        return await asyncio.to_thread(adapter.systemone, payload)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("decision systemone failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
```

In `router/openai_router.py`, add the import after the `traces_router` import (line ~39):

```python
from router.decision_router import router as decision_router
```

Then add, alongside the other `app.include_router(...)` calls (search `app.include_router(traces_router)`):

```python
app.include_router(decision_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_decision_router.py -v --basetemp=C:\Users\fred\AppData\Local\Temp\pytest-mm`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the router import smoke check**

Run: `.venv\Scripts\python.exe -c "import router.openai_router; print('router import ok')"`
Expected: `router import ok`.

- [ ] **Step 6: Commit**

```bash
git add router/decision_router.py router/openai_router.py tests/test_decision_router.py
git commit -m "feat: add /v1/systemone decision endpoint"
```

---

### Task 4: Register the models and document

**Files:**
- Modify: `config/models.yaml` (append three entries)
- Modify: `README.md` (add a `### Decision Models` section; add `/v1/systemone` to the endpoint list)
- Modify: `PROGRESS.md` (append)
- Modify: `TASK.md` (mark done)

**Interfaces:**
- Consumes: Tasks 1-3 (capability + endpoint).
- Produces: no code; configuration + docs.

- [ ] **Step 1: Register models in `config/models.yaml`**

Append under the top-level `models:` list:

```yaml
- name: tev1:0.8b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities:
  - decision
  - tools
  - thinking
  - chat
  estimated_vram_mb: null
  context_length: 262144
- name: tev1:4b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities:
  - decision
  - tools
  - thinking
  - chat
  estimated_vram_mb: null
  context_length: 262144
- name: nimble:9b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities:
  - decision
  - tools
  - thinking
  - chat
  estimated_vram_mb: null
  context_length: 262144
```

- [ ] **Step 2: Add `/v1/systemone` to the README endpoint list**

Under `### OpenAI-Compatible API`, after the `POST /v1/rerank` entry:

```markdown
- `POST /v1/systemone` — Ollama decision models (choice/score questions over a state)
```

- [ ] **Step 3: Add a README section**

After the `### Rerank API` section:

```markdown
### Decision Models (`POST /v1/systemone`)

Ollama decision models (e.g. `nimble:9b`, `tev1:0.8b`, `tev1:4b`) answer structured questions
about a `state`. AetherMesh forwards the request to the Ollama worker that owns the model.

```bash
curl http://localhost:8001/v1/systemone \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "nimble:9b", "state": "Book a flight to Tokyo",
       "questions": {"intent": {"type": "choice", "instructions": "What does the user want?",
       "criteria": {"book": "book travel", "cancel": "cancel", "other": null}}}}'
```

Response:

```json
{"model": "nimble:9b",
 "answers": {"intent": {"type": "choice", "choice": "book", "probabilities": {"book": 0.99}, "confidence": 0.95}},
 "usage": {"input_tokens": 171, "output_tokens": 1}}
```

`questions` accepts 1–64 fields; each has `type` = `choice` | `noul` | `score`, `instructions`,
and (for `choice`) a `criteria` object mapping options to descriptions (or `null`), or (for `score`)
a `criteria` array of descriptions. Register decision models in `config/models.yaml` (or the Dashboard
**Models** tab) with the `decision` capability. Non-Ollama providers return `501`.
```

- [ ] **Step 4: Append PROGRESS.md and update TASK.md**

Append a dated entry to `PROGRESS.md` describing the new endpoint, capability, adapter method, and model registrations. In `TASK.md`, mark the Decision models section items complete.

- [ ] **Step 5: Verify the endpoint end-to-end (manual, if services running)**

With the router restarted, run a real decision request:

```bash
curl http://localhost:8001/v1/systemone -H "Content-Type: application/json" ^
  -d "{\"model\":\"nimble:9b\",\"state\":\"Book a flight to Tokyo\",\"questions\":{\"intent\":{\"type\":\"choice\",\"instructions\":\"What does the user want?\",\"criteria\":{\"book\":\"book travel\",\"cancel\":\"cancel\",\"other\":null}}}}"
```

Expected: `answers.intent.choice == "book"`.

- [ ] **Step 6: Commit**

```bash
git add config/models.yaml README.md PROGRESS.md TASK.md
git commit -m "docs: register decision models and document /v1/systemone"
```

---

## Self-Review

- **Spec coverage:** §2 contract → Global Constraints + Task 3 tests; §3.1 capability → Task 1; §3.2 Dashboard list → Task 1; §3.3 models.yaml → Task 4; §3.4 adapter → Task 2; §3.5 route → Task 3; §3.6 provider resolution → Task 3; §5 validation/errors → Task 3 tests; §6 tests → each task; §7 docs → Task 4.
- **Placeholder scan:** none — every code/test step is complete; Task 4 PROGRESS text is descriptive (documentation content, not implementation).
- **Type consistency:** `OllamaAdapter.systemone(payload) -> dict` (Task 2) is what Task 3 calls; `provider_router.resolve_provider` / `adapter` signatures match Task 3 usage; `decision` literal is consistent across Tasks 1, 3, 4.
- **Review Focus:** empty/>64 questions + upstream errors → Task 3 `test_upstream_error_is_preserved`; `state` object/array pass-through → Task 3 `test_routes_to_ollama_adapter` (adapter receives the raw `payload`); non-Ollama → Task 3 `test_unsupported_provider_returns_501`; connection failure → Task 2 `test_systemone_raises_on_error` + Task 3 generic 502 handler.
