# Vision Support for AetherMesh

Date: 2025-07-06
Status: Draft
Author: AetherMesh Dev

## Problem

白龍馬 (White Dragon Horse) voice client needs vision/image analysis. Currently AetherMesh
supports ASR (speech-to-text) and TTS (text-to-speech), but cannot route vision requests
to Vision Language Models (VLMs).

## Current State

AetherMesh already has the vision infrastructure in place:

- **Content block parsing** (`runtime/tools/content_blocks.py`): Bidirectional conversion
  between Anthropic `image` blocks and OpenAI `image_url` blocks.
- **Vision capability detection** (`runtime/orchestration/capabilities.py`): Automatically
  detects `image_url`/`image`/`input_image` blocks and adds `vision` to required capabilities.
- **Vision-aware routing** (`runtime/orchestration/routing_engine.py`):
  `CAPABILITY_PROVIDER_SCORES` includes vision scores (Gemini 98, OpenAI 95, Ollama 80);
  `_local_model_fallback()` finds vision-capable local Ollama models.
- **Ollama adapter** (`providers/ollama_adapter.py`): Converts `image_url` base64 data to
  Ollama `images` array field.
- **Gemini adapter** (`providers/gemini_adapter.py`): Converts `image_url` to Gemini
  `inline_data` format.
- **Model config** (`config/models.yaml`): Some models (gemma4, qwen3.6 variants) already
  tagged with `capabilities: [vision]`.

## Design

### Phase 1: Config-First Vision (zero core code changes)

Goal: Get vision working with configuration changes only.

#### 1a. GPU Worker Isolation

Current GPU memory:
- RTX 5090 (GPU 0): 32 GB, ~30 GB used → no room for VLM
- RTX 4070 Ti (GPU 1): 16 GB, ~8 GB used → 7.5 GB free → fits 7B VLM (~4.5 GB)

All VLM traffic routes to port 11435 (GPU 1) via an Ollama worker:

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve &
```

#### 1b. Model Configuration

`config/cluster.yaml` — add `role: vision` to GPU 1 worker.

The VLM model is NOT statically added to `config/models.yaml`. Instead, the routing engine
dynamically checks `settings.vision_model()` as an additional fallback candidate when no
registered model supports vision. This avoids requiring users to edit YAML files — setting
`AIIH_VISION_MODEL` and pulling the model in Ollama is sufficient.

Optionally, users may also add the model to `config/models.yaml` for explicit worker binding
and advanced options (num_ctx, etc.). If both exist, the static entry takes precedence.

#### 1c. Environment Variable

`.env` / `.env.example`:

```ini
# Vision / VLM
AIIH_VISION_MODEL=qwen2.5-vl:7b         # preferred local VLM (ollama)
AIIH_VISION_FALLBACK=any                  # off | openai | gemini | any (Gemini first, then OpenAI)
```

#### 1d. Settings Update

`config/settings.py` — add `vision_model()` accessor that reads `AIIH_VISION_MODEL`.

### Phase 2: Cloud Fallback (routing enhancement)

Goal: When no local VLM is available, auto-route to cloud vision providers.

#### 2a. RoutingEngine Modification

File: `runtime/orchestration/routing_engine.py`

Two changes:

**Change 1 — Add dynamic VLM fallback in `_local_model_fallback()`** (line 552):

After iterating `registry_models`, check `settings.vision_model()` as additional candidate:

```python
def _local_model_fallback(self, required_capabilities, registry_models):
    required = set(required_capabilities or ["chat"])
    # 1. Try registered models first
    for model in registry_models:
        if str(model.get("provider", "ollama")).lower() != "ollama":
            continue
        if not model.get("worker_bindings"):
            continue
        if required.issubset(set(model.get("capabilities", []))):
            return model
    # 2. Try AIIH_VISION_MODEL as dynamic fallback
    vision_model = settings.vision_model()
    if "vision" in required and vision_model:
        # Check if model exists on any worker
        for worker in local_workers:
            if _check_worker_model(worker["port"], vision_model):
                return {"name": vision_model, "provider": "ollama",
                        "capabilities": ["chat", "vision"],
                        "worker_bindings": [
                            {"node_id": "node-01", "port": worker["port"]}
                        ]}
    # 3. Last resort: any Ollama model
    for model in registry_models:
        if str(model.get("provider", "ollama")).lower() == "ollama" and model.get("worker_bindings"):
            return model
    return None
```

**Change 2 — Add cloud fallback** (after line 448, when local fallback returns None):

```python
if "vision" in required_capabilities and not fallback:
    allowed = settings.VISION_FALLBACK  # "any", "openai", "gemini", or "off"
    if allowed == "off":
        rules_applied.append("vision_cloud_fallback_disabled")
    else:
        cloud_vision_scores = CAPABILITY_PROVIDER_SCORES["vision"]
        providers_order = sorted(cloud_vision_scores, key=lambda p: cloud_vision_scores[p], reverse=True)
        for provider_name in providers_order:
            if provider_name not in CLOUD_PROVIDERS:
                continue
            if allowed not in ("any", provider_name):
                continue
            if not _provider_has_api_key(provider_name):
                continue
            if provider_name not in VISION_CLOUD_MODELS:
                continue
            model_name = VISION_CLOUD_MODELS[provider_name]
            best = RouteCandidate(
                provider=provider_name,
                model=model_name,
                score=CAPABILITY_PROVIDER_SCORES["vision"].get(provider_name, 50),
                reason=f"vision_cloud_fallback_{provider_name}",
            )
            rules_applied.append(f"vision_cloud_fallback {provider_name}/{model_name}")
            break
```

#### 2b. Cloud Provider Vision Model Mapping

```python
VISION_CLOUD_MODELS = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
}
```

### Phase 3: Testing

File: `tests/test_vision.py`

```
test_vision_chat_local         — image_url → local VLM → description
test_vision_routes_to_vlm      — text-only model rerouted to AIIH_VISION_MODEL
test_vision_streaming          — streaming chat with image
test_vision_fallback           — no local VLM → cloud fallback
test_vision_gemini_adapter     — Gemini adapter image conversion
test_vision_anthropic_format   — Anthropic image block conversion
```

### Data Flow

```
白龍馬 POST /v1/audio/transcriptions  (speech→text)
     → "Describe what you see"
     → 白龍馬 captures photo, sends:
POST /v1/chat/completions
{
  "model": "gpt-4o",                ← 白龍馬選任意 model
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
     → openai_handler.py
       → capabilities.py: detects image_url → requires vision
       → routing_engine.py: selected model lacks vision
         → _local_model_fallback(): finds AIIH_VISION_MODEL
         → routes to port 11435 (GPU 1, 4070 Ti)
       → ollama_adapter.py: image_url → base64 → Ollama images[]
       → Ollama VLM processes image
       → response → TTS reads aloud
```

### Files Changed

| File | Change Type | Phase |
|------|-------------|-------|
| `config/cluster.yaml` | Add role:vision to GPU 1 worker (optional) | 1 |
| `.env.example` | Add AIIH_VISION_MODEL, AIIH_VISION_FALLBACK | 1 |
| `config/settings.py` | Add `vision_model()` accessor | 1 |
| `runtime/orchestration/routing_engine.py` | Add cloud fallback for vision | 2 |
| `tests/test_vision.py` | New test file | 3 |
| `README.md` | Document vision setup | 3 |

### Non-Goals

- No new provider adapter (Ollama/Gemini adapters already handle vision)
- No new API endpoint (/v1/chat/completions is the unified interface)
- No response content block reconstruction changes
- No VLM Provider Adapter abstraction (can be added in future as Phase 4)
