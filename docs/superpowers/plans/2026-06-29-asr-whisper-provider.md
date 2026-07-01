# ASR (faster-whisper) Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI-compatible `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` powered by local faster-whisper on GPU.

**Architecture:** 3-layer stack mirroring TTS — `router/audio_router.py` → `runtime/orchestration/provider_router.py` → `providers/asr_base.py` / `providers/faster_whisper_adapter.py`.

**Tech Stack:** faster-whisper (CTranslate2), soundfile, tempfile

---

## File Structure

**Create:**
- `providers/asr_base.py` — ABC + error class
- `providers/faster_whisper_adapter.py` — concrete adapter
- `tests/test_asr_base.py` — 8 tests
- `tests/test_faster_whisper_adapter.py` — 10 tests
- `requirements-asr.txt` — dependency manifest

**Modify:**
- `config/settings.py` — add 4 ASR fields
- `config/models.yaml` — add whisper model entry
- `.env.example` — add ASR section
- `.env` — add ASR settings
- `runtime/orchestration/provider_router.py` — asr() branch + singleton + ROUTE_PREFIXES
- `router/audio_router.py` — add transcription + translation endpoints
- `router/openai_router.py` — conditional audio_router include
- `tests/test_audio_router.py` — add ASR endpoint tests

---

### Task 1: ASR Base Class + Error

**Files:**
- Create: `providers/asr_base.py`
- Create: `tests/test_asr_base.py`

- [ ] **Step 1: Write `providers/asr_base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ASRProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class ASRProviderAdapter(ABC):
    provider_name: str = "asr_base"

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        ...
```

- [ ] **Step 2: Write `tests/test_asr_base.py`**

```python
from __future__ import annotations

from providers.asr_base import ASRProviderAdapter, ASRProviderError


def test_asr_provider_error_is_runtime_error() -> None:
    assert issubclass(ASRProviderError, RuntimeError)


def test_asr_provider_error_default_status() -> None:
    err = ASRProviderError("boom")
    assert err.status_code == 500


def test_asr_provider_error_custom_status() -> None:
    err = ASRProviderError("not found", status_code=404)
    assert err.status_code == 404


def test_asr_provider_adapter_is_abstract() -> None:
    import pytest
    with pytest.raises(TypeError):
        ASRProviderAdapter()  # type: ignore[abstract]


def test_abstract_methods_exist() -> None:
    import inspect
    methods = [
        m for m in ASRProviderAdapter.__dict__
        if not m.startswith("_") or m == "__init__"
    ]
    assert "transcribe" in inspect.getsource(ASRProviderAdapter.transcribe)


def test_provider_name_default() -> None:
    assert ASRProviderAdapter.provider_name == "asr_base"
```

- [ ] **Step 3: Run tests to verify they fail (no module yet)**

Run: `pytest tests/test_asr_base.py -x -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 4: Create file and run tests to verify they pass**

Run: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_asr_base.py -x -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add providers/asr_base.py tests/test_asr_base.py
git commit -m "feat(asr): add ASR provider base class + error"
```

---

### Task 2: Settings + Config Registration

**Files:**
- Modify: `config/settings.py` (after line 140)
- Modify: `config/models.yaml` (append to models list)
- Modify: `.env.example` (append ASR section)

- [ ] **Step 1: Add ASR settings to `config/settings.py`**

After `tts_dtype` field (line 141), add:

```python
    asr_enabled: bool = field(default_factory=lambda: _env_bool("AIIH_ASR_ENABLED", "false"))
    asr_model: str = field(default_factory=lambda: os.getenv("AIIH_ASR_MODEL", "large-v3"))
    asr_device: str = field(default_factory=lambda: os.getenv("AIIH_ASR_DEVICE", "cuda"))
    asr_compute_type: str = field(default_factory=lambda: os.getenv("AIIH_ASR_COMPUTE_TYPE", "float16"))
```

- [ ] **Step 2: Add model entry to `config/models.yaml`**

Append to the models list (before the `# ── Local TTS ──` section or after the TTS entry):

```yaml
  # ── Local ASR ──────────────────────────────────────────────
  - name: whisper-large-v3
    provider: asr
    worker_ports: []
    capabilities: [audio]
```

- [ ] **Step 3: Add ASR section to `.env.example`**

After the TTS section (after `AIIH_TTS_DTYPE=fp32`), add:

```
# ── Local ASR (faster-whisper) ──────────────────────────────
# [control] Enable local speech-to-text via faster-whisper on GPU
# Requires: pip install -r requirements-asr.txt
AIIH_ASR_ENABLED=false
AIIH_ASR_MODEL=large-v3
AIIH_ASR_DEVICE=cuda
AIIH_ASR_COMPUTE_TYPE=float16
```

- [ ] **Step 4: Add ASR settings to `.env`**

Append to `.env`:

```
AIIH_ASR_ENABLED=true
AIIH_ASR_MODEL=large-v3
AIIH_ASR_DEVICE=cuda
AIIH_ASR_COMPUTE_TYPE=float16
```

- [ ] **Step 5: Commit**

```bash
git add config/settings.py config/models.yaml .env.example .env
git commit -m "feat(asr): add ASR config settings + model registration"
```

---

### Task 3: faster-whisper Adapter

**Files:**
- Create: `providers/faster_whisper_adapter.py`
- Create: `tests/test_faster_whisper_adapter.py`
- Create: `requirements-asr.txt`

- [ ] **Step 1: Write `requirements-asr.txt`**

```
faster-whisper>=1.1.0
```

- [ ] **Step 2: Write the test file `tests/test_faster_whisper_adapter.py`**

```python
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.asr_base import ASRProviderError


@pytest.fixture(autouse=True)
def _mock_asr_deps() -> MagicMock:
    """Mock faster_whisper, soundfile, torch to prevent GPU imports."""
    with patch.dict("sys.modules"):
        import sys

        # Mock faster_whisper
        fw = MagicMock()
        fw.WhisperModel = MagicMock()
        fake_model = MagicMock()
        fake_segment = MagicMock()
        fake_segment.text = "hello world"
        fake_segment.start = 0.0
        fake_segment.end = 2.5
        fake_model.transcribe.return_value = [fake_segment], None  # info
        fw.WhisperModel.return_value = fake_model
        sys.modules["faster_whisper"] = fw

        # Mock soundfile
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        sys.modules["soundfile"] = mock_sf

        # Mock torch (imported by adapter for audio loading fallback)
        sys.modules["torch"] = MagicMock()

        yield fake_model


@pytest.fixture
def adapter(_mock_asr_deps: MagicMock) -> MagicMock:
    from providers.faster_whisper_adapter import FasterWhisperAdapter
    return FasterWhisperAdapter(
        model_name="large-v3",
        device="cpu",
        compute_type="int8",
    )


class TestFasterWhisperAdapterInit:
    def test_provider_name(self, adapter: MagicMock) -> None:
        assert adapter.provider_name == "faster_whisper"

    def test_model_loaded(self, adapter: MagicMock) -> None:
        assert adapter._model is not None


class TestFasterWhisperAdapterTranscribe:
    def test_transcribe_returns_text(self, adapter: MagicMock) -> None:
        result = adapter.transcribe(audio=b"fake-wav-data")
        assert isinstance(result, dict)
        assert "text" in result
        assert result["text"] == "hello world"

    def test_translate_task(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        adapter.transcribe(audio=b"fake", task="translate")
        _mock_asr_deps.transcribe.assert_called_once()
        _, kwargs = _mock_asr_deps.transcribe.call_args
        assert kwargs.get("task") == "translate"

    def test_language_param(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        adapter.transcribe(audio=b"fake", language="zh")
        _, kwargs = _mock_asr_deps.transcribe.call_args
        assert kwargs.get("language") == "zh"

    def test_temperature_param(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        adapter.transcribe(audio=b"fake", temperature=0.5)
        _, kwargs = _mock_asr_deps.transcribe.call_args
        assert kwargs.get("temperature") == 0.5

    def test_prompt_param(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        adapter.transcribe(audio=b"fake", prompt="context hint")
        _, kwargs = _mock_asr_deps.transcribe.call_args
        assert kwargs.get("initial_prompt") == "context hint"

    def test_auto_language_when_empty(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        adapter.transcribe(audio=b"fake", language="")
        _, kwargs = _mock_asr_deps.transcribe.call_args
        # When language is empty, it should not be passed (auto-detect)
        assert kwargs.get("language") is None

    def test_health_check(self, adapter: MagicMock) -> None:
        health = adapter.health_check()
        assert health["provider"] == "faster_whisper"
        assert health["model_loaded"] is True
        assert "device" in health


class TestFasterWhisperAdapterErrors:
    def test_corrupted_audio_raises(self, adapter: MagicMock, _mock_asr_deps: MagicMock) -> None:
        _mock_asr_deps.transcribe.side_effect = ValueError("corrupted audio")
        with pytest.raises(ASRProviderError) as exc:
            adapter.transcribe(audio=b"trash")
        assert exc.value.status_code == 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_faster_whisper_adapter.py -x -v`
Expected: FAIL (ModuleNotFoundError for faster_whisper_adapter)

- [ ] **Step 4: Write `providers/faster_whisper_adapter.py`**

```python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from providers.asr_base import ASRProviderAdapter, ASRProviderError

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


class FasterWhisperAdapter(ASRProviderAdapter):
    provider_name = "faster_whisper"

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = self._load_model()

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel
        return WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(audio)
            tmp.flush()
            tmp.close()

            transcribe_kwargs: dict[str, Any] = {
                "task": task,
                "temperature": temperature,
            }
            if language:
                transcribe_kwargs["language"] = language
            if prompt:
                transcribe_kwargs["initial_prompt"] = prompt

            segments, info = self._model.transcribe(
                tmp.name,
                **transcribe_kwargs,
            )
            text = " ".join(seg.text for seg in segments)
            return {"text": text}
        except ValueError as e:
            raise ASRProviderError(str(e), status_code=400) from e
        except Exception as e:
            raise ASRProviderError(str(e), status_code=500) from e
        finally:
            try:
                os.unlink(tmp.name)
            except (OSError, AttributeError):
                pass

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_faster_whisper_adapter.py -x -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add providers/faster_whisper_adapter.py tests/test_faster_whisper_adapter.py requirements-asr.txt
git commit -m "feat(asr): add faster-whisper adapter"
```

---

### Task 4: Provider Router + Audio Router Endpoints

**Files:**
- Modify: `runtime/orchestration/provider_router.py`
- Modify: `router/audio_router.py`
- Add ASR tests to: `tests/test_audio_router.py`

- [ ] **Step 1: Wire ASR into `runtime/orchestration/provider_router.py`**

Add after the XTTSAdapter import block:

```python
try:
    from providers.faster_whisper_adapter import FasterWhisperAdapter
except ImportError:
    FasterWhisperAdapter = None  # type: ignore[assignment, misc]
```

Add before the `ROUTE_PREFIXES` dict:

```python
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

In the `adapter()` function, add after the xtts branch:

```python
    if provider == "asr":
        if FasterWhisperAdapter is None:
            raise ValueError("ASR adapter not available (faster-whisper not installed)")
        return _get_asr_adapter()
```

In `resolve_provider()`, add `"asr"` to the hinted_providers list:

```python
            if hinted_provider in ("openai", "gemini", "nvidia_nim", "ollama_cloud", "xtts", "asr"):
```

Add to `ROUTE_PREFIXES`:

```python
    "asr/": "asr",
```

- [ ] **Step 2: Add transcription + translation endpoints to `router/audio_router.py`**

After the existing TTS endpoints and before `# ── ASR ──` section, add:

```python
# ── ASR ──────────────────────────────────────────────────────

def _resolve_asr_adapter():
    if not settings.asr_enabled:
        raise HTTPException(status_code=503, detail="ASR is not enabled")
    return get_adapter("asr")


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_asr_adapter()
    try:
        return adapter.transcribe(
            audio=audio_data,
            task="transcribe",
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )
    except ASRProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_asr_adapter()
    try:
        return adapter.transcribe(
            audio=audio_data,
            task="translate",
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )
    except ASRProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
```

Also add the ASRProviderError import at the top of `router/audio_router.py`:

```python
from providers.asr_base import ASRProviderError
```

- [ ] **Step 3: Write ASR endpoint tests in `tests/test_audio_router.py`**

Append after the existing voice CRUD tests:

```python
class TestASR:
    @pytest.fixture(autouse=True)
    def _enable_asr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "asr_enabled", True)

    @pytest.fixture
    def mock_asr(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        fake = MagicMock()
        fake.transcribe.return_value = {"text": "hello world"}
        from runtime.orchestration import provider_router
        monkeypatch.setattr(provider_router, "_get_asr_adapter", lambda: fake)
        monkeypatch.setattr(provider_router, "FasterWhisperAdapter", MagicMock())
        return fake

    async def test_transcribe_success(self, client: AsyncClient, mock_asr: MagicMock) -> None:
        resp = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "hello world"

    async def test_translate_success(self, client: AsyncClient, mock_asr: MagicMock) -> None:
        resp = await client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "hello world"

    async def test_transcribe_missing_file(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-large-v3"},
        )
        assert resp.status_code == 422

    async def test_asr_disabled(self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "asr_enabled", False)
        resp = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"data", "audio/wav")},
        )
        assert resp.status_code == 503

    async def test_transcribe_adapter_error(self, client: AsyncClient, mock_asr: MagicMock) -> None:
        from providers.asr_base import ASRProviderError
        mock_asr.transcribe.side_effect = ASRProviderError("bad audio", status_code=400)
        resp = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"bad", "audio/wav")},
        )
        assert resp.status_code == 400
        assert "bad audio" in resp.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_audio_router.py -x -v -k "TestASR"`
Expected: 5 passed

Run full test suite: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_asr_base.py tests/test_faster_whisper_adapter.py tests/test_audio_router.py -v`
Expected: 21 passed total

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/provider_router.py router/audio_router.py tests/test_audio_router.py
git commit -m "feat(asr): add ASR provider router + audio endpoints"
```

---

### Task 5: Wire into openai_router + Final Assembly

**Files:**
- Modify: `router/openai_router.py`

- [ ] **Step 1: Update `router/openai_router.py` conditional include**

Change:

```python
if settings.tts_enabled:
```

To:

```python
if settings.tts_enabled or settings.asr_enabled:
```

- [ ] **Step 2: Run full ASR test suite**

Run: `$env:PYTHONPATH = "D:\Ai\AetherMesh"; pytest tests/test_asr_base.py tests/test_faster_whisper_adapter.py tests/test_audio_router.py -v`
Expected: 21 passed

- [ ] **Step 3: Commit**

```bash
git add router/openai_router.py
git commit -m "feat(asr): include audio router when ASR is enabled"
```

---

## Verification

After all tasks:

```bash
# Full ASR test suite
pytest tests/test_asr_base.py tests/test_faster_whisper_adapter.py tests/test_audio_router.py -v

# Full project test suite
pytest tests/ -x --ignore=tests/test_dashboard_auth.py -v
```

Expected: 296+21 = 317+ passed (pre-existing failures unchanged)
