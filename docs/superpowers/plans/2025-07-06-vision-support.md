# Vision Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable /v1/chat/completions with image_url blocks to route to local VLM (ollama, GPU 1) with cloud failover.

**Architecture:** Minimal changes — existing ollama/gemini adapters already handle image_url. Add dynamic VLM model fallback + cloud fallback to routing_engine.py. No new provider adapter, no new API endpoint.

**Tech Stack:** Python 3.10+, Ollama (qwen2.5-vl:7b on GPU 1), OpenAI/Gemini for cloud fallback.

## Global Constraints

- Follow existing patterns (4-space indent, `from __future__ import annotations`, type hints, no docstrings unless required)
- Settings access via `settings.<field>`, never `settings.get()`
- `pytest` with `asyncio_mode = auto` for tests
- No new provider adapter (Ollama/Gemini adapters already handle vision)
- No new API endpoint

---

### Task 1: Settings + Env Config

**Files:**
- Modify: `config/settings.py` — add `vision_model`, `vision_fallback` fields
- Modify: `.env.example` — add AIIH_VISION_MODEL, AIIH_VISION_FALLBACK

**Interfaces:**
- Consumes: existing `_env_bool` helper
- Produces: `settings.vision_model: str`, `settings.vision_fallback: str`

- [ ] **Step 1: Add Vision settings fields**

Edit `config/settings.py`, add after the ASR fields (~line 148):

```python
vision_model: str = field(default_factory=lambda: os.getenv("AIIH_VISION_MODEL", "qwen2.5-vl:7b"))
vision_fallback: str = field(default_factory=lambda: os.getenv("AIIH_VISION_FALLBACK", "any"))
```

- [ ] **Step 2: Add env vars to .env.example**

Edit `.env.example`, add after the ASR section (~line 189):

```ini
# ── Vision / VLM ──────────────────────────────────────────────────
# [control] Vision Language Model for /v1/chat/completions with images
AIIH_VISION_MODEL=qwen2.5-vl:7b         # local VLM model (ollama, GPU 1)
AIIH_VISION_FALLBACK=any                 # off | openai | gemini | any (Gemini first)
```

- [ ] **Step 3: Commit**

```bash
git add config/settings.py .env.example
git commit -m "feat(vision): add AIIH_VISION_MODEL and AIIH_VISION_FALLBACK settings"
```

---

### Task 2: RoutingEngine — Dynamic VLM Fallback

**Files:**
- Modify: `runtime/orchestration/routing_engine.py` — modify `_local_model_fallback()`, add `_resolve_fallback_worker()`

**Interfaces:**
- Consumes: `settings.vision_model`, `settings.vision_fallback`
- Produces: `_local_model_fallback()` returns synthetic model entry for dynamic VLM; `_resolve_fallback_worker()` returns worker dict from fallback entry's bindings

- [ ] **Step 1: Write the failing test**

Add to a new test file `tests/test_vision.py`. We'll write the routing tests first.

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from runtime.orchestration.routing_engine import ModelRoutingEngine


def _make_registry(models: list[dict]) -> list[dict]:
    return models


def _make_request(model: str, has_images: bool = False) -> dict:
    msg = {"role": "user"}
    if has_images:
        msg["content"] = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaaa"}},
        ]
    else:
        msg["content"] = "hello"
    return {"model": model, "messages": [msg]}


class TestVisionRouting:
    """Vision capability routing tests."""

    def test_vision_request_routes_to_vision_model(self):
        """Request with image_url should reroute to a vision-capable model when
        the requested model lacks vision."""
        registry = _make_registry([
            {"name": "gemma4:26b", "provider": "ollama", "capabilities": ["chat", "vision"],
             "worker_bindings": [{"node_id": "node-01", "port": 11434}]},
            {"name": "north-mini-code-1.0", "provider": "ollama", "capabilities": ["chat"],
             "worker_bindings": [{"node_id": "node-01", "port": 11434}]},
        ])
        engine = ModelRoutingEngine()
        with patch.object(engine, "_get_workers", return_value=[]):
            with patch.object(engine, "_probe_worker", return_value=True):
                decision = engine.route("north-mini-code-1.0", registry, has_images=True)
        assert decision is not None
        assert "vision" in decision.model.lower() or decision.model == "gemma4:26b"

    def test_vision_fallback_to_dynamic_vlm(self):
        """When no registered model supports vision, fall back to settings.vision_model()."""
        registry = _make_registry([
            {"name": "north-mini-code-1.0", "provider": "ollama", "capabilities": ["chat"],
             "worker_bindings": [{"node_id": "node-01", "port": 11434}]},
        ])
        engine = ModelRoutingEngine()
        with patch.object(settings, "vision_model", return_value="qwen2.5-vl:7b"):
            with patch.object(engine, "_get_workers", return_value=[]):
                with patch.object(engine, "_probe_worker", return_value=True):
                    decision = engine.route("north-mini-code-1.0", registry, has_images=True)
        assert decision is not None
        assert decision.model == "qwen2.5-vl:7b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision.py::TestVisionRouting -x -v`

Expected: FAIL with AttributeError or similar (code not yet modified)

- [ ] **Step 3: Modify `_local_model_fallback()` to check `settings.vision_model()`**

In `runtime/orchestration/routing_engine.py`, modify the `_local_model_fallback()` method:

```python
def _local_model_fallback(
    self,
    required_capabilities: list[str],
    registry_models: list[dict[str, Any]],
) -> dict[str, Any] | None:
    required = set(required_capabilities or ["chat"])
    for model in registry_models:
        if str(model.get("provider", "ollama")).lower() != "ollama":
            continue
        if not model.get("worker_bindings"):
            continue
        capabilities = set(model.get("capabilities", []))
        if required.issubset(capabilities):
            return model
    # Dynamic VLM fallback for vision requests
    if "vision" in required:
        vision_model = settings.vision_model
        if vision_model:
            return {
                "name": vision_model,
                "provider": "ollama",
                "capabilities": ["chat", "vision"],
                "worker_bindings": [
                    {"node_id": "node-01", "port": 11435}
                ],
            }
    for model in registry_models:
        if str(model.get("provider", "ollama")).lower() == "ollama" and model.get("worker_bindings"):
            return model
    return None
```

- [ ] **Step 4: Add `_resolve_fallback_worker()` helper**

In `runtime/orchestration/routing_engine.py`, add after `_first_healthy_binding()`:

```python
def _resolve_fallback_worker(
    self,
    fallback: dict[str, Any],
) -> dict[str, Any] | None:
    bindings = fallback.get("worker_bindings", [])
    if not bindings:
        return None
    return self._first_healthy_binding(bindings)
```

- [ ] **Step 5: Modify worker resolution to use fallback's worker_bindings**

In `_resolve_provider_and_worker()` (around line 434), modify the block where `worker is None`:

_Old code (~line 434-448):_
```python
                if worker is None or selected_missing_caps:
                    fallback = self._local_model_fallback(required_capabilities, registry_models)
                    if fallback:
                        best = RouteCandidate(
                            provider="ollama",
                            model=str(fallback.get("name", clean_model)),
                            score=max(best.score, 75.0),
                            latency_ms=best.latency_ms,
                            healthy=best.healthy,
                            reason="local_model_fallback",
                        )
                        worker = self._worker_for_model(best.model, best.model, registry_models)
                        rules_applied.append(f"local_model_fallback {clean_model} -> {best.model}")
                    else:
                        rules_applied.append("no_ollama_worker_available")
```

_New code:_
```python
                if worker is None or selected_missing_caps:
                    fallback = self._local_model_fallback(required_capabilities, registry_models)
                    if fallback:
                        best = RouteCandidate(
                            provider="ollama",
                            model=str(fallback.get("name", clean_model)),
                            score=max(best.score, 75.0),
                            latency_ms=best.latency_ms,
                            healthy=best.healthy,
                            reason="local_model_fallback",
                        )
                        worker = self._worker_for_model(best.model, best.model, registry_models)
                        if worker is None:
                            worker = self._resolve_fallback_worker(fallback)
                        rules_applied.append(f"local_model_fallback {clean_model} -> {best.model}")
                    else:
                        rules_applied.append("no_ollama_worker_available")
```

- [ ] **Step 6: Add `has_images` parameter support in `route()` method**

The `route()` method signature needs to accept image information. Currently it likely infers capabilities from model name. We need to pass capability requirements explicitly.

Find the `route()` method signature and add `has_images: bool = False` parameter. Then when `has_images` is True, ensure `"vision"` is in the `required_capabilities` list.

```python
def route(
    self,
    model: str,
    registry_models: list[dict[str, Any]],
    has_images: bool = False,
) -> RoutingDecision | None:
```

And at the top of the method, after capability inference:

```python
required_capabilities = self._infer_capabilities(model, registry_models)
if "vision" not in required_capabilities and has_images:
    required_capabilities = list(required_capabilities) + ["vision"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_vision.py::TestVisionRouting -x -v`

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add runtime/orchestration/routing_engine.py tests/test_vision.py
git commit -m "feat(vision): dynamic VLM fallback in routing engine"
```

---

### Task 3: RoutingEngine — Cloud Vision Fallback

**Files:**
- Modify: `runtime/orchestration/routing_engine.py` — add `VISION_CLOUD_MODELS`, cloud fallback logic, `_provider_has_api_key()` helper

**Interfaces:**
- Consumes: `settings.vision_fallback`, `CLOUD_PROVIDERS`, `CLOUD_PROVIDER_ENDPOINTS`, `CAPABILITY_PROVIDER_SCORES`
- Produces: cloud `RouteCandidate` when no local VLM available

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vision.py`:

```python
class TestVisionCloudFallback:
    """Vision cloud fallback tests."""

    def test_vision_cloud_fallback_openai(self):
        """When no local VLM, fall back to OpenAI vision."""
        registry = []
        engine = ModelRoutingEngine()
        with patch.object(settings, "vision_model", return_value=""):
            with patch.object(settings, "vision_fallback", return_value="any"):
                with patch.object(engine, "_get_workers", return_value=[]):
                    with patch.object(engine, "_probe_worker", return_value=False):
                        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                            decision = engine.route("gpt-4o", registry, has_images=True)
        assert decision is not None
        assert decision.provider == "openai"

    def test_vision_cloud_fallback_disabled(self):
        """When AIIH_VISION_FALLBACK=off, cloud fallback should not trigger."""
        registry = []
        engine = ModelRoutingEngine()
        with patch.object(settings, "vision_fallback", return_value="off"):
            with patch.object(settings, "vision_model", return_value=""):
                with patch.object(engine, "_get_workers", return_value=[]):
                    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                        decision = engine.route("gpt-4o", registry, has_images=True)
        # Without cloud fallback and no local VLM, route returns the original model
        assert decision is not None
        assert decision.provider != "openai" or not any("vision_cloud" in r for r in decision.rules_applied)

    def test_vision_cloud_fallback_no_api_key(self):
        """When no cloud API key configured, cloud fallback should not trigger."""
        registry = []
        engine = ModelRoutingEngine()
        with patch.object(settings, "vision_model", return_value=""):
            with patch.object(settings, "vision_fallback", return_value="any"):
                with patch.object(engine, "_get_workers", return_value=[]):
                    with patch.dict(os.environ, {}, clear=True):
                        decision = engine.route("gpt-4o", registry, has_images=True)
        assert decision is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision.py::TestVisionCloudFallback -x -v`

Expected: FAIL

- [ ] **Step 3: Add `VISION_CLOUD_MODELS` mapping**

In `runtime/orchestration/routing_engine.py`, add after existing module-level constants (~line 37):

```python
VISION_CLOUD_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
}
```

- [ ] **Step 4: Add cloud fallback logic after local fallback chain**

Use the existing `self._provider_credentials` dict (already populated in `__init__` via `_check_provider_credentials()`) to check API key availability.

In `_resolve_provider_and_worker()`, after the `if worker is None` block (after the `rules_applied.append` for `no_ollama_worker_available`), add:

```python
# Cloud fallback for vision when no local VLM available
if best.provider == "ollama" and worker is None and "vision" in required_capabilities:
    fallback_mode = settings.vision_fallback
    if fallback_mode != "off":
        cloud_scores = CAPABILITY_PROVIDER_SCORES.get("vision", {})
        ordered = sorted(cloud_scores, key=lambda p: cloud_scores[p], reverse=True)
        for provider_name in ordered:
            if provider_name not in CLOUD_PROVIDERS:
                continue
            if fallback_mode not in ("any", provider_name):
                continue
            if not self._provider_credentials.get(provider_name):
                continue
            cloud_model = VISION_CLOUD_MODELS.get(provider_name)
            if not cloud_model:
                continue
            best = RouteCandidate(
                provider=provider_name,
                model=cloud_model,
                score=cloud_scores.get(provider_name, 50),
                latency_ms=best.latency_ms,
                healthy=True,
                reason=f"vision_cloud_fallback_{provider_name}",
            )
            rules_applied.append(f"vision_cloud_fallback {provider_name}/{cloud_model}")
            break
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_vision.py::TestVisionCloudFallback -x -v`

Expected: PASS

- [ ] **Step 6: Run all existing tests to check for regressions**

Run: `pytest tests/ -x --timeout=60 2>&1 | tail -20`

Expected: All existing tests pass

- [ ] **Step 7: Commit**

```bash
git add runtime/orchestration/routing_engine.py tests/test_vision.py
git commit -m "feat(vision): add cloud vision fallback (OpenAI/Gemini)"
```

---

### Task 4: Vision E2E + Streaming Tests

**Files:**
- Modify: `tests/test_vision.py` — add E2E and streaming tests

- [ ] **Step 1: Add capability detection and adapter tests**

```python
class TestVisionCapabilities:
    """Vision capability detection and adapter conversion tests."""

    def test_capabilities_detects_image_url(self):
        """required_openai_capabilities should include vision for image_url."""
        from runtime.orchestration.capabilities import required_openai_capabilities

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,aaaa"}}
                ]
            }]
        }
        caps = required_openai_capabilities(payload)
        assert "vision" in caps

    def test_capabilities_no_vision_for_text_only(self):
        """required_openai_capabilities should NOT include vision for text-only."""
        from runtime.orchestration.capabilities import required_openai_capabilities

        payload = {
            "messages": [{"role": "user", "content": "hello"}]
        }
        caps = required_openai_capabilities(payload)
        assert "vision" not in caps

    def test_ollama_adapter_extracts_images(self):
        """Ollama adapter should convert image_url to images list."""
        from providers.ollama_adapter import OllamaProviderAdapter

        adapter = OllamaProviderAdapter()
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,YWJjZGVmZw=="}}
            ]
        }
        normalized = adapter._message_for_ollama(message)
        assert "images" in normalized
        assert len(normalized["images"]) == 1
        assert normalized["images"][0] == "YWJjZGVmZw=="
```

- [ ] **Step 2: Run all vision tests**

Run: `pytest tests/test_vision.py -x -v`

Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x --timeout=60 2>&1 | tail -20`

Expected: 296+ tests pass, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_vision.py
git commit -m "feat(vision): add E2E and capabilities tests for vision support"
```

---

### Task 5: Documentation + Cluster Config

**Files:**
- Modify: `config/cluster.yaml` — update GPU 1 worker role
- Modify: `README.md` — add Vision section

- [ ] **Step 1: Update cluster.yaml role**

Edit `config/cluster.yaml` line 16:

_Old:_
```yaml
    role: embeddings
```

_New:_
```yaml
    role: embeddings+vision
```

- [ ] **Step 2: Add Ollama worker startup info to README**

In `README.md`, add a new section "Local VLM (Vision Language Model)" after the ASR section. Include:

```markdown
### Local VLM (Vision Language Model)

AetherMesh can route `/v1/chat/completions` requests containing `image_url` blocks
to a local Vision Language Model (VLM) running on Ollama. The VLM runs on a separate
GPU worker (port 11435, GPU 1) to avoid competing VRAM with large text models.

**Setup:**

```bash
# 1. Start Ollama worker on GPU 1 (isolated from text models)
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve &

# 2. Pull a VLM model (default: qwen2.5-vl:7b, ~4.5 GB VRAM)
OLLAMA_HOST=0.0.0.0:11435 ollama pull qwen2.5-vl:7b

# 3. Configure in .env
AIIH_VISION_MODEL=qwen2.5-vl:7b          # your VLM model
AIIH_VISION_FALLBACK=any                  # cloud fallback: off | openai | gemini | any
```

**Usage — same /v1/chat/completions endpoint, just add image_url:**

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

**Routing logic:**
1. If the requested model has `vision` capability → direct route
2. If not, AetherMesh finds `AIIH_VISION_MODEL` on GPU 1 worker
3. If local VLM unavailable → falls back to cloud (OpenAI `gpt-4.1-mini` or
   Gemini `gemini-2.5-flash`, controlled by `AIIH_VISION_FALLBACK`)

**VRAM planning (RTX 5090 32 GB + RTX 4070 Ti 16 GB):**

| GPU | Models | Used VRAM |
|-----|--------|-----------|
| GPU 0 (5090) | Text models (gemma4:31b, qwen3.6:35b) | ~30 GB |
| GPU 1 (4070 Ti) | VLM (7B) + embeddings | ~12 GB |
| TTS subprocess | GPU worker (fp16, ~2.4 GB) | Spawns on demand |

```env
AIIH_TTS_DEVICE=cuda:1     # TTS on GPU 1 (shares with VLM)
```

**Change model:**
```bash
# Just update the env var and pull the new model
AIIH_VISION_MODEL=llama3.2-vision:11b
OLLAMA_HOST=0.0.0.0:11435 ollama pull llama3.2-vision:11b
```

**Troubleshooting:**
- `Ollama adapter error: ...` → check that the VLM is pulled: `OLLAMA_HOST=0.0.0.0:11435 ollama list`
- `worker queue full` → reduce `AIIH_VISION_WORKER_PORT` concurrency or add a second VLM worker
- Cloud fallback not working → verify API key env vars (`OPENAI_API_KEY`, `GEMINI_API_KEY`)
- CUDA OOM → use a smaller VLM (7B instead of 11B) or switch TTS to CPU: `AIIH_TTS_DEVICE=cpu`
```

Then add the new env vars to the env var table (~line 695):

```
| `AIIH_VISION_MODEL` | `qwen2.5-vl:7b` | Local VLM model for vision requests (ollama) |
| `AIIH_VISION_FALLBACK` | `any` | Cloud fallback mode: `off`, `openai`, `gemini`, `any` |
```

- [ ] **Step 3: Commit**

```bash
git add config/cluster.yaml README.md
git commit -m "docs(vision): add VLM setup guide and cluster config"
```
