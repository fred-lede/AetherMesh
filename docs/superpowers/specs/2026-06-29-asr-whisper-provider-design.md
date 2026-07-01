# ASR (faster-whisper) Provider Design

**Date:** 2026-06-29
**Status:** Approved
**Author:** AetherMesh Agent

## 1. Goal

Add an OpenAI-compatible `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` endpoint to AetherMesh, powered by `faster-whisper` running locally on the 4070Ti Super GPU.

## 2. Architecture

Same 3-layer pattern as the TTS provider:

```
router/audio_router.py          ← new POST /v1/audio/transcriptions + translations
       ↓
runtime/orchestration/provider_router.py   ← asr() branch
       ↓
providers/asr_base.py                    ← abstract base class
providers/faster_whisper_adapter.py      ← concrete implementation
```

### 2.1 Data Flow

1. Client sends multipart audio file to `POST /v1/audio/transcriptions`
2. `audio_router.py` reads the file, forwards to `_resolve_asr_adapter()`
3. `provider_router.asr()` returns the singleton `FasterWhisperAdapter`
4. Adapter loads `WhisperModel(large-v3, device=cuda, compute_type=float16)` on first call
5. `model.transcribe(audio_array, task=task, language=lang, ...)` runs on GPU
6. Returns `{"text": "..."}` JSON response

## 3. Provider Layer

### 3.1 `providers/asr_base.py`

```python
class ASRProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code


class ASRProviderAdapter(ABC):
    provider_name: str = "asr_base"

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",       # "transcribe" | "translate"
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        ...
```

- `transcribe()` is synchronous (run via `asyncio.to_thread` in the async route)
- Returns `{"text": str}` for `json` format
- `ASRProviderError` carries `status_code` for HTTP mapping

### 3.2 `providers/faster_whisper_adapter.py`

```python
class FasterWhisperAdapter(ASRProviderAdapter):
    provider_name = "faster_whisper"

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self._model = self._load_model(model_name, device, compute_type)

    def transcribe(self, audio: bytes, ...) -> dict[str, Any]:
        # Write bytes to temp WAV -> read with soundfile -> numpy array
        # model.transcribe(audio_array, task=task, language=lang, ...)
        # Return {"text": segments.text}
```

**Key implementation details:**
- Model loaded lazily once (singleton pattern in `provider_router`)
- `compute_type="float16"` uses ~3GB VRAM for `large-v3`
- Audio bytes saved to temp file → `faster-whisper` reads the path directly (handles any format via internal ffmpeg)
- `model.transcribe()` returns segment generator — concatenate all segment texts
- `language=""` means auto-detect (Whisper native)
- `task="translate"` handles `POST /v1/audio/translations`

## 4. Router Layer

Extend `router/audio_router.py` with two new endpoints:

```
POST /v1/audio/transcriptions
  Body: multipart/form-data
    - file: UploadFile (required, any audio format ffmpeg can decode)
    - model: str (default: "whisper-large-v3")
    - language: str (optional, ISO-639-1)
    - prompt: str (optional)
    - temperature: float (default: 0.0)
    - response_format: str (default: "json")
  Response 200:
    {"text": "transcribed text"}

POST /v1/audio/translations
  Body: same as above (language param is ignored, output is always English)
  Response 200:
    {"text": "english translation text"}
```

Both endpoints:
- Are `async` but call `adapter.transcribe()` via `asyncio.to_thread()` since it's CPU/GPU-bound
- Handle `ASRProviderError` → `HTTPException`
- Use `_resolve_asr_adapter()` that checks `settings.asr_enabled` (returns 503 if disabled)
- Audio format conversion handled by `soundfile` (supports WAV, FLAC, MP3 via ffmpeg-bridge capability of `soundfile` or `ffmpeg` fallback)

**OpenAI compatibility:**
- `model` field is accepted but currently ignored (only one model: `whisper-large-v3`)
- `response_format` is accepted but only `"json"` is supported (returns `{"text": "..."}`)
- `temperature` maps to `WhisperModel.transcribe(temperature=...)`
- `prompt` maps to `initial_prompt` parameter

## 5. Config Changes

### 5.1 `config/settings.py`

```python
asr_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_ASR_ENABLED", "false"))
asr_model: str = field(default_factory=lambda: os.getenv("AIIH_ASR_MODEL", "large-v3"))
asr_device: str = field(default_factory=lambda: os.getenv("AIIH_ASR_DEVICE", "cuda"))
asr_compute_type: str = field(default_factory=lambda: os.getenv("AIIH_ASR_COMPUTE_TYPE", "float16"))
```

### 5.2 `config/models.yaml`

```yaml
  - name: whisper-large-v3
    provider: asr
    worker_ports: []
    capabilities: [audio]
```

### 5.3 `.env.example` / `.env`

```
# ── Local ASR (faster-whisper) ────────────────────────────────
AIIH_ASR_ENABLED=false
AIIH_ASR_MODEL=large-v3
AIIH_ASR_DEVICE=cuda
AIIH_ASR_COMPUTE_TYPE=float16
```

### 5.4 `requirements-asr.txt`

```
faster-whisper>=1.1.0
soundfile      # already in requirements.txt, listed for clarity
```

## 6. Provider Router Integration

### 6.1 `runtime/orchestration/provider_router.py`

```python
try:
    from providers.faster_whisper_adapter import FasterWhisperAdapter
except ImportError:
    FasterWhisperAdapter = None

# In adapter():
    if provider == "asr":
        if FasterWhisperAdapter is None:
            raise ValueError("ASR adapter not available (faster-whisper not installed)")
        return _get_asr_adapter()

# In resolve_provider(), add "asr" to the shortcut list
# Add to ROUTE_PREFIXES: "asr/": "asr"

_asr_adapter: Any | None = None

def _get_asr_adapter() -> Any:
    global _asr_adapter
    if _asr_adapter is None and settings.asr_enabled:
        _asr_adapter = FasterWhisperAdapter(
            model_name=settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
    return _asr_adapter
```

### 6.2 `router/openai_router.py`

```python
if settings.tts_enabled:
    from router.audio_router import router as audio_router
    app.include_router(audio_router)
```

Change to:

```python
if settings.tts_enabled or settings.asr_enabled:
    from router.audio_router import router as audio_router
    app.include_router(audio_router)
```

## 7. Dependencies

`requirements-asr.txt`:
```
faster-whisper>=1.1.0
```

`faster-whisper` depends on `ctranslate2`, `huggingface-hub`, `numpy`, `tokenizers`. Audio loading goes through `soundfile` (already present in `requirements.txt`).

Total additional install size: ~150MB (CTranslate2 wheels + whisper model).

## 8. Security Considerations

- Audio files are written to a temp directory and deleted after transcription
- Model runs entirely local — no audio data leaves the machine
- `AIIH_API_KEY` authentication applies to all `/v1/audio/*` endpoints (already enforced by middleware)
- Input audio is decoded with `soundfile` which handles malformed files gracefully

## 9. Testing Strategy

### 9.1 `tests/test_asr_base.py` (8 tests)

- Test that `ASRProviderAdapter` cannot be instantiated
- Test that `ASRProviderError` is a `RuntimeError` with `status_code`
- Test all `transcribe()` method signatures are abstract
- Test `provider_name` default
- Test `ASRProviderError` with default status (500) and custom status

### 9.2 `tests/test_faster_whisper_adapter.py` (10 tests)

- Mock `faster_whisper.WhisperModel` entirely
- Mock `soundfile` for audio reads
- Test init: `provider_name`, model loaded, health check
- Test `transcribe()` returns `{"text": "..."}`
- Test `transcribe()` with `task="translate"` passes correct arg
- Test `transcribe()` with `language="zh"` passes correct arg
- Test `transcribe()` with `temperature=0.5` passes correct arg
- Test error handling when audio file is corrupted
- Test auto-detect language when `language=""`

### 9.3 `tests/test_audio_router.py` (add ~6 ASR tests)

- Test `POST /v1/audio/transcriptions` success (mock adapter)
- Test `POST /v1/audio/translations` success (mock adapter)
- Test missing file returns 422
- Test adapter error returns correct HTTP status
- Test ASR disabled returns 503
- Test ASR response format is valid JSON

## 10. Future Extensions

- Support `response_format = "verbose_json"` with word-level timestamps
- Add `POST /v1/models` to expose `whisper-large-v3` in the model list
- VAD (Voice Activity Detection) pre-processing to skip silence
- Speaker diarization (WhisperX integration)
- Model hot-swap for smaller models (`medium`, `small`, `base`, `tiny`)
