# Local XTTS-v2 TTS Provider — Design Spec

**Date:** 2026-06-29
**Status:** Draft
**GPU:** RTX 4070Ti Super (12GB VRAM)
**Model:** XTTS-v2 via Coqui `TTS` library

---

## 1. Goal

Add local **text-to-speech** capability to AetherMesh using XTTS-v2 on the
4070Ti Super GPU, exposed via an **OpenAI-compatible `/v1/audio/speech`**
REST API with **pre-registered voice cloning** support.

Future TTS backends (Fish Speech, ChatTTS, etc.) should be swappable through a
common abstraction layer.

---

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│              router/audio_router.py           │
│  /v1/audio/speech                            │
│  /v1/voices/* (CRUD for cloned voices)        │
│  /v1/models (append TTS models to list)       │
└─────────────┬────────────────────────────────┘
              │
┌─────────────▼────────────────────────────────┐
│  runtime/orchestration/provider_router.py    │
│  adapter("xtts") → XTTSAdapter               │
└─────────────┬────────────────────────────────┘
              │
┌─────────────▼────────────────────────────────┐
│           providers/tts_base.py              │
│         TTSProviderAdapter (ABC)             │
└─────────────┬────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
┌───▼──────────┐  ┌────▼───────────┐
│ xtts_adapter │  │ fish_adapter   │
│ (XTTS-v2)    │  │ (future)       │
│ GPU · 4-5GB  │  │                │
└──────────────┘  └────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility |
|-------|---------------|
| `router/` | HTTP request parsing, OpenAI format adaptation, audio format conversion (WAV/MP3/OPUS), voice management routes |
| `providers/` | Model loading, inference, voice embedding extraction/storage |
| `runtime/` | Provider resolution via existing `provider_router.py`, GPU resource tracking via `VRAMScheduler` |

---

## 3. API Surface

### 3.1 `POST /v1/audio/speech`

OpenAI-compatible shape with AetherMesh extensions:

```json
{
  "model": "xtts-v2",
  "input": "您好，這是語音合成測試",
  "voice": "my-voice-id",
  "response_format": "wav",
  "speed": 1.0,
  "language": "zh-tw"
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | `string` | `"xtts-v2"` | Supports future models like `"fish-speech"` |
| `input` | `string` | (required) | Text to synthesize |
| `voice` | `string` | (required) | Pre-registered voice ID (UUID) from `/v1/voices` |
| `response_format` | `string` | `"wav"` | One of `wav, mp3, opus, flac`. `mp3`/`opus`/`flac` require `ffmpeg`. |
| `speed` | `number` | `1.0` | Playback speed multiplier (0.5–2.0) |
| `language` | `string` | auto-detect | XTTS-v2 supports: `en, zh-tw, zh-cn, ja, ko, fr, de, es, it, pt, pl, tr, ru, nl, ar, cs, hu, hi` |

**Success response (200):** Binary audio bytes with `Content-Type: audio/<format>`.

**Error response (400/422):** Standard JSON error body.

### 3.2 `POST /v1/voices`

Register a new voice from a reference audio sample (multipart form-data).

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | `string` | yes | Human-readable label, e.g., `"fred-voice"` |
| `file` | `file` | yes | Audio file (WAV/MP3), 3–15 seconds, mono preferred |
| `language` | `string` | no | Language hint for embedding extraction |

Returns:
```json
{
  "voice_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "fred-voice",
  "language": "zh-tw",
  "created_at": "2026-06-29T12:00:00Z",
  "duration_seconds": 8.2
}
```

### 3.3 `GET /v1/voices`

List all registered voices.

### 3.4 `GET /v1/voices/{voice_id}`

Get details for a single voice (name, language, created_at, duration).

### 3.5 `DELETE /v1/voices/{voice_id}`

Delete a cloned voice and its stored embedding.

### 3.6 `GET /v1/models` (integration with existing endpoint)

Appends the TTS model(s) to the existing `/v1/models` response so clients
can discover TTS capability.

---

## 4. Provider Layer

### 4.1 `providers/tts_base.py` — `TTSProviderAdapter(ABC)`

```python
class TTSProviderAdapter(ABC):
    provider_name: str = "tts_base"

    @abstractmethod
    def tts(self, payload: dict) -> bytes:
        """Synthesize speech. Returns raw audio bytes (WAV PCM 24000Hz)."""
        ...

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """Return list of registered voice metadata dicts."""
        ...

    @abstractmethod
    def register_voice(
        self, name: str, audio_data: bytes,
        language: str = "", content_type: str = "audio/wav"
    ) -> dict:
        """Register a new voice from reference audio. Returns voice metadata."""
        ...

    @abstractmethod
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a registered voice."""
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """Return model load status, VRAM usage, registered voice count."""
        ...
```

### 4.2 `providers/xtts_adapter.py` — XTTS-v2 Implementation

- **Dependencies:** `TTS` (Coqui), `soundfile`, `scipy`
- **Model init (on process start):**
  1. Load XTTS-v2 with `torch.float16` on `cuda:0`
  2. Expected VRAM usage: ~4–5 GB
  3. Register VRAM footprint with `VRAMScheduler` as a synthetic `GPUResource`
- **Voice storage:** `data/voices/` directory
  - Each voice: a subdirectory `{voice_id}/` containing:
    - `speaker_embedding.pt` — cloned speaker latents
    - `meta.json` — `{name, language, created_at, duration_seconds}`
- **`tts()` flow:**
  1. Look up `voice` ID → load `speaker_embedding.pt`
  2. Call `model.tts(text=text, speaker_embedding=embedding, language=lang)`
  3. Return WAV bytes (24000 Hz, mono, PCM 16-bit)
- **Error states:**
  - Voice ID not found → `KeyError` → router returns 404
  - Model not loaded / GPU OOM → `ProviderError` with `status_code=503`
  - Text too long (>200 chars) → returns 422; long text requires chunking

### 4.3 GPU Resource Registration

At adapter init, register with VRAMScheduler:

```python
vram_scheduler.update_gpu(GPUResource(
    gpu_id="xtts-v2",
    node_id="localhost",
    worker_port=0,  # TTS is in-process, not a remote worker
    vram=VRAMProfile(total_mb=5120, used_mb=0, free_mb=5120),
    model_loaded="xtts-v2",
    queue_depth=0,
    healthy=True,
    tier=0,
    metadata={"type": "tts", "provider": "xtts"}
))
```

This makes TTS visible in health metrics and dashboard.

---

## 5. Router Layer

### 5.1 `router/audio_router.py`

New file implementing the API surface from §3.

- **`POST /v1/audio/speech`:**
  1. Parse and validate request body
  2. Resolve provider via `provider_router.adapter("xtts")`
  3. Build internal payload dict
  4. Call `adapter.tts(payload)` → WAV bytes
  5. Convert to requested `response_format` via `ffmpeg` subprocess (or return raw WAV)
  6. Return `Response(content=audio_bytes, media_type="audio/<format>")`

- **Format conversion:** Use `ffmpeg` via subprocess if available:
  - WAV → MP3: `ffmpeg -i pipe:0 -f mp3 -b:a 192k pipe:1`
  - WAV → OPUS: `ffmpeg -i pipe:0 -f opus -b:a 96k pipe:1`
  - WAV → FLAC: `ffmpeg -i pipe:0 -f flac pipe:1`
  - Fallback: if no `ffmpeg`, only WAV format is supported.

- **`POST /v1/voices`:** Accept multipart upload, validate audio, call `adapter.register_voice()`
- **`GET /v1/voices`:** Call `adapter.list_voices()`
- **`DELETE /v1/voices/{voice_id}`:** Call `adapter.delete_voice()`

### 5.2 Model Registration

Append `"xtts-v2"` to the model list returned by `GET /v1/models` so clients
can discover TTS capability through the standard models endpoint.

---

## 6. Config Changes

### `config/settings.py` — new fields

| Field | Env Var | Default | Notes |
|-------|---------|---------|-------|
| `tts_model_name` | `AIIH_TTS_MODEL` | `"tts_models/multilingual/multi-dataset/xtts_v2"` | XTTS-v2 model ID |
| `tts_device` | `AIIH_TTS_DEVICE` | `"cuda:0"` | GPU device for TTS |
| `tts_voices_dir` | `AIIH_TTS_VOICES_DIR` | `"data/voices"` | Cloned voice storage |
| `tts_models_dir` | `AIIH_TTS_MODELS_DIR` | `"data/tts"` | XTTS-v2 model cache |
| `tts_enabled` | `AIIH_TTS_ENABLED` | `false` | Master toggle (opt-in) |

### `config/models.yaml` — new entry

```yaml
xtts-v2:
  capabilities: [audio]
  provider: xtts
```

### `providers/registry.py`

Register a `"xtts"` provider entry with `Capability.AUDIO`.

### `runtime/orchestration/provider_router.py`

- Add `"xtts/"` to `ROUTE_PREFIXES` (for explicit model prefix routing)
- In `adapter()`, handle `provider == "xtts"` → return `XTTSAdapter` singleton

---

## 7. Dependencies

| Package | Purpose | Size | Notes |
|---------|---------|------|-------|
| `TTS` | XTTS-v2 model + inference | ~300 MB | Coqui TTS package (includes PyTorch) |
| `soundfile` | WAV read/write | small | Already often installed with TTS |
| `scipy` | WAV output fallback | ~15 MB | Already likely in env |
| `ffmpeg` | Audio format conversion | ~100 MB | System dep (optional). Use `ffmpeg-python` or raw subprocess |

**Total new runtime dependency weight:** ~315 MB (mostly Coqui TTS + PyTorch).

---

## 8. Security Considerations

- **API key auth:** All `/v1/audio/speech` and `/v1/voices/*` routes use the same
  auth middleware as other endpoints (env var + DB API keys).
- **Audio file validation:** `POST /v1/voices` validates:
  - File extension (`.wav`, `.mp3`, `.ogg`, `.flac`)
  - Max file size (10 MB)
  - Duration (3–15 seconds recommended)
  - MIME type check
- **Voice data isolation:** Each voice is stored as files on disk, accessible
  only by voice_id. No path traversal risk if IDs are UUIDs.

---

## 9. Testing Strategy

- **Unit tests** in `tests/test_xtts_adapter.py`:
  - Mock or monkeypatch the `TTS` model at module import
  - Test `tts()` with mocked model returning sample audio tensor
  - Test voice register / list / delete with temp directory
  - Test error states (voice not found, model not loaded)
- **Unit tests** in `tests/test_audio_router.py`:
  - Test request validation (missing fields, bad voice ID)
  - Test response content types
  - Test auth integration
- **GPU tests** (manual/skip on CI):
  - Actual model load and inference on 4070Ti Super
  - VRAM usage verification
  - Voice cloning quality check

---

## 10. Future Extensions

| Feature | Backend | Notes |
|---------|---------|-------|
| Fish Speech | `fish_speech_adapter.py` | Higher quality, supports more languages |
| ChatTTS | `chat_tts_adapter.py` | Conversational TTS |
| CosyVoice | `cosy_voice_adapter.py` | Alibaba, good Chinese TTS |
| SSE streaming TTS | Streaming chunked audio | `/v1/audio/speech` with `stream=true` |
| Voice cloning from mic | Real-time capture | Dashboard integration |
| Long text chunking | Split >200 chars, merge audio | Required for production use |
