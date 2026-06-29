# Local TTS Provider (XTTS-v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local XTTS-v2 text-to-speech to AetherMesh with voice cloning, exposed via OpenAI-compatible `/v1/audio/speech` and `/v1/voices/*` CRUD API.

**Architecture:** `TTSProviderAdapter(ABC)` in `providers/tts_base.py` abstracts any TTS backend. `XTTSAdapter` wraps the Coqui XTTS-v2 model (FP16, GPU). `router/audio_router.py` handles OpenAI format conversion and voice management.

**Tech Stack:** Coqui TTS, PyTorch (CUDA), soundfile, ffmpeg (optional), pytest

---

### Task 1: Dependency Registration

**Files:**
- Modify: `pyproject.toml`
- Modify: `config/settings.py`
- Modify: `config/models.yaml`
- Modify: `providers/registry.py`

- [ ] **Step 1: Add optional TTS dependency to pyproject.toml**

Add under `[project.optional-dependencies]`:

```toml
tts = [
    "TTS",
    "soundfile",
]
```

- [ ] **Step 2: Add TTS config fields to config/settings.py**

```python
@dataclass(slots=True)
class Settings:
    # ... existing fields ...

    # TTS configuration
    tts_enabled: bool = field(
        default_factory=lambda: _env_bool("AIIH_TTS_ENABLED", False)
    )
    tts_model_name: str = field(
        default_factory=lambda: os.getenv(
            "AIIH_TTS_MODEL",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )
    )
    tts_device: str = field(
        default_factory=lambda: os.getenv("AIIH_TTS_DEVICE", "cuda:0")
    )
    tts_voices_dir: str = field(
        default_factory=lambda: os.getenv("AIIH_TTS_VOICES_DIR", "data/voices")
    )
    tts_models_dir: str | None = field(
        default_factory=lambda: os.getenv("AIIH_TTS_MODELS_DIR")
    )
```

- [ ] **Step 3: Add xtts-v2 model to config/models.yaml**

Append to the model list:

```yaml
xtts-v2:
  capabilities: [audio]
  provider: xtts
```

- [ ] **Step 4: Register xtts provider in providers/registry.py**

After the existing `register_providers()` function or at module level, add:

```python
if settings.tts_enabled:
    provider_capability_registry.register(
        ProviderCapabilityEntry(
            name="xtts",
            capabilities={Capability.AUDIO},
            healthy=True,
            latency_ms=0,
            requires_key=False,
        )
    )
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config/settings.py config/models.yaml providers/registry.py
git commit -m "feat(tts): add TTS config, model registry entry, and capability registration"
```

---

### Task 2: TTS Provider Abstraction

**Files:**
- Create: `providers/tts_base.py`
- Create: `tests/test_tts_base.py`

- [ ] **Step 1: Write the test for TTSProviderAdapter ABC**

Create `tests/test_tts_base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from providers.tts_base import TTSProviderAdapter, TTSProviderError


class TestTTSProviderAdapterInterface:
    """Verify the ABC contract: all abstract methods exist."""

    def test_is_abstract(self) -> None:
        assert issubclass(TTSProviderAdapter, ABC)

    def test_abstract_methods(self) -> None:
        """Subclasses must implement: tts, list_voices, register_voice, delete_voice, health_check."""
        expected = {"tts", "list_voices", "register_voice", "delete_voice", "health_check"}
        abstract = {
            m for m in TTSProviderAdapter.__abstractmethods__  # type: ignore[attr-defined]
        }
        assert abstract == expected, f"Missing abstracts: {expected - abstract}"

    def test_tts_signature(self) -> None:
        from inspect import signature
        sig = signature(TTSProviderAdapter.tts)
        assert "payload" in sig.parameters

    def test_list_voices_signature(self) -> None:
        from inspect import signature
        sig = signature(TTSProviderAdapter.list_voices)
        # no required params beyond self
        assert len(sig.parameters) == 1

    def test_register_voice_signature(self) -> None:
        from inspect import signature
        sig = signature(TTSProviderAdapter.register_voice)
        for param in ("name", "audio_data"):
            assert param in sig.parameters

    def test_delete_voice_signature(self) -> None:
        from inspect import signature
        sig = signature(TTSProviderAdapter.delete_voice)
        assert "voice_id" in sig.parameters

    def test_provider_name_default(self) -> None:
        assert TTSProviderAdapter.provider_name == "tts_base"


class TestTTSProviderError:
    def test_default_status_code(self) -> None:
        err = TTSProviderError("oops")
        assert err.status_code == 500

    def test_custom_status_code(self) -> None:
        err = TTSProviderError("not found", status_code=404)
        assert err.status_code == 404

    def test_is_runtime_error(self) -> None:
        assert issubclass(TTSProviderError, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_base.py -v`
Expected: ModuleNotFoundError or ImportError for `providers.tts_base`

- [ ] **Step 3: Write minimal TTSProviderAdapter**

Create `providers/tts_base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TTSProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class TTSProviderAdapter(ABC):
    provider_name: str = "tts_base"

    @abstractmethod
    def tts(self, payload: dict[str, Any]) -> bytes:
        """Synthesize speech from text. Returns WAV PCM 24000Hz mono bytes."""
        ...

    @abstractmethod
    def list_voices(self) -> list[dict[str, Any]]:
        """Return list of registered voice metadata dicts."""
        ...

    @abstractmethod
    def register_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Register a new voice from reference audio bytes."""
        ...

    @abstractmethod
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a registered voice. Returns True if deleted, False if not found."""
        ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return provider health status."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_base.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add providers/tts_base.py tests/test_tts_base.py
git commit -m "feat(tts): add TTSProviderAdapter ABC with tests"
```

---

### Task 3: XTTS-v2 Adapter Implementation

**Files:**
- Create: `providers/xtts_adapter.py`
- Create: `tests/test_xtts_adapter.py`

This task creates the real XTTS-v2 adapter. Tests use monkeypatching/mocking so they run without a GPU or TTS library installed.

- [ ] **Step 1: Write the test file**

Create `tests/test_xtts_adapter.py`. Include a test for GPU VRAM registration:

```python
def test_gpu_vram_registration(adapter: XTTSAdapter) -> None:
    """XTTSAdapter registers itself with VRAMScheduler on init."""
    from runtime.gpu.vram_scheduler import vram_scheduler
    resource = vram_scheduler._gpus.get("xtts-v2")
    assert resource is not None
    assert resource.gpu_id == "xtts-v2"
    assert resource.model_loaded == "xtts-v2"
    assert resource.vram.total_mb > 0
```

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from providers.tts_base import TTSProviderError
from providers.xtts_adapter import XTTSAdapter

MODULE = "providers.xtts_adapter"


@pytest.fixture
def mock_tts_api() -> MagicMock:
    """Mock the entire TTS.api module to prevent GPU import."""
    with patch.dict("sys.modules"):
        fake_tts = MagicMock()
        fake_tts.tts.return_value = np.zeros((24000,), dtype=np.float32)

        tts_module = MagicMock()
        tts_module.TTS.return_value = fake_tts

        # Insert fakes *before* importing xtts_adapter
        import sys
        sys.modules["TTS"] = tts_module
        sys.modules["TTS.api"] = tts_module
        sys.modules["TTS.utils"] = MagicMock()
        sys.modules["TTS.utils.generic_utils"] = MagicMock()

        yield fake_tts


@pytest.fixture
def mock_soundfile() -> MagicMock:
    with patch(f"{MODULE}.sf") as mock_sf:
        mock_sf.write.return_value = None
        yield mock_sf


@pytest.fixture
def adapter(mock_tts_api: MagicMock, mock_soundfile: MagicMock, tmp_path: Path) -> XTTSAdapter:
    return XTTSAdapter(
        model_name="test-model",
        device="cpu",
        voices_dir=str(tmp_path / "voices"),
    )


class TestXTTSAdapterInit:
    def test_provider_name(self, adapter: XTTSAdapter) -> None:
        assert adapter.provider_name == "xtts"

    def test_voices_dir_created(self, adapter: XTTSAdapter, tmp_path: Path) -> None:
        assert (tmp_path / "voices").is_dir()

    def test_model_loaded(self, adapter: XTTSAdapter) -> None:
        assert adapter._model is not None


class TestXTTSAdapterTTS:
    def test_tts_returns_bytes(self, adapter: XTTSAdapter, monkeypatch) -> None:
        # Register a fake voice first
        adapter.register_voice = MagicMock(return_value={"voice_id": "test-id", "name": "test"})  # type: ignore[method-assign]
        adapter._load_embedding = MagicMock(return_value=np.zeros((1, 256), dtype=np.float32))  # type: ignore[method-assign]

        result = adapter.tts({
            "voice": "test-id",
            "input": "Hello world",
            "language": "en",
        })
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_tts_missing_voice_raises(self, adapter: XTTSAdapter) -> None:
        adapter._load_embedding = MagicMock(side_effect=FileNotFoundError)  # type: ignore[method-assign]
        with pytest.raises(TTSProviderError) as exc:
            adapter.tts({"voice": "nonexistent", "input": "hello"})
        assert exc.value.status_code == 404

    def test_tts_applies_speed(self, adapter: XTTSAdapter, monkeypatch) -> None:
        applied = []

        def fake_speed(data: bytes, speed: float) -> bytes:
            applied.append(speed)
            return data

        monkeypatch.setattr(adapter, "_apply_speed", fake_speed)
        monkeypatch.setattr(adapter, "_load_embedding", MagicMock(return_value=np.zeros((1, 256))))
        adapter.tts({"voice": "x", "input": "hi", "speed": 1.5})
        assert applied == [1.5]


class TestXTTSAdapterVoiceCRUD:
    def test_register_and_list(self, adapter: XTTSAdapter, tmp_path: Path) -> None:
        meta = adapter.register_voice(
            name="test-voice",
            audio_data=b"fake-wav-data",
            language="en",
        )
        assert "voice_id" in meta
        assert meta["name"] == "test-voice"
        assert meta["language"] == "en"

        voices = adapter.list_voices()
        assert any(v["voice_id"] == meta["voice_id"] for v in voices)

    def test_delete_voice(self, adapter: XTTSAdapter) -> None:
        meta = adapter.register_voice(name="del-me", audio_data=b"data")
        assert adapter.delete_voice(meta["voice_id"]) is True
        assert adapter.delete_voice("nonexistent") is False

    def test_health_check(self, adapter: XTTSAdapter) -> None:
        health = adapter.health_check()
        assert health["provider"] == "xtts"
        assert "model_loaded" in health
        assert "device" in health
        assert "voices_count" in health
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xtts_adapter.py -v`
Expected: ImportError for `providers.xtts_adapter`

- [ ] **Step 3: Write minimal XTTSAdapter**

Create `providers/xtts_adapter.py`:

```python
from __future__ import annotations

import io
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from providers.tts_base import TTSProviderAdapter, TTSProviderError

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


class XTTSAdapter(TTSProviderAdapter):
    provider_name = "xtts"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda:0",
        voices_dir: str = "data/voices",
        models_dir: str | None = None,
    ) -> None:
        self._device = device
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._model = self._load_model(model_name, models_dir)

    def _load_model(self, model_name: str, models_dir: str | None) -> Any:
        from TTS.api import TTS
        tts = TTS(model_name=model_name, model_dir=models_dir, progress_bar=False)
        tts.to(self._device)
        return tts

    def _voice_path(self, voice_id: str) -> Path:
        return self._voices_dir / voice_id

    def _load_embedding(self, voice_id: str) -> tuple[Any, Any]:
        vp = self._voice_path(voice_id)
        emb_path = vp / "speaker_embedding.pt"
        if not emb_path.exists():
            raise TTSProviderError(f"Voice {voice_id} not found", status_code=404)
        import torch
        data = torch.load(emb_path, map_location=self._device, weights_only=True)
        return data["gpt_cond_latent"], data["speaker_embedding"]

    def tts(self, payload: dict[str, Any]) -> bytes:
        voice_id = payload["voice"]
        text = payload["input"]
        language = payload.get("language", "en")
        speed = payload.get("speed", 1.0)

        gpt_cond, speaker_embed = self._load_embedding(voice_id)
        wav: np.ndarray = self._model.tts(
            text=text,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=speaker_embed,
            language=language,
        )
        buffer = io.BytesIO()
        if sf is not None:
            sf.write(buffer, wav, 24000, format="WAV", subtype="PCM_16")
        else:
            from scipy.io.wavfile import write as wav_write
            wav_write(buffer, 24000, wav)
        audio_bytes = buffer.getvalue()

        if speed != 1.0:
            audio_bytes = self._apply_speed(audio_bytes, speed)

        return audio_bytes

    def _apply_speed(self, wav_bytes: bytes, speed: float) -> bytes:
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-filter:a", f"atempo={speed}",
                 "-f", "wav", "pipe:1"],
                input=wav_bytes, capture_output=True, check=True, timeout=30,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return wav_bytes

    def register_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        voice_id = str(uuid4())
        vp = self._voice_path(voice_id)
        vp.mkdir(parents=True, exist_ok=True)

        ref_path = vp / "reference.wav"
        ref_path.write_bytes(audio_data)

        gpt_cond, speaker_embed = self._model.get_conditioning_latents(
            audio_path=str(ref_path)
        )
        import torch
        torch.save(
            {"gpt_cond_latent": gpt_cond, "speaker_embedding": speaker_embed},
            str(vp / "speaker_embedding.pt"),
        )

        duration = self._get_audio_duration(audio_data)
        meta = {
            "voice_id": voice_id,
            "name": name,
            "language": language,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
        }
        (vp / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def _get_audio_duration(self, audio_data: bytes) -> float:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp = f.name
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp],
                capture_output=True, text=True, check=True, timeout=10,
            )
            Path(tmp).unlink(missing_ok=True)
            return round(float(result.stdout.strip()), 2)
        except Exception:
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp = f.name
            info = sf.info(tmp)
            Path(tmp).unlink(missing_ok=True)
            return round(info.duration, 2)

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for entry in sorted(self._voices_dir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                voices.append(json.loads(meta_path.read_text()))
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        vp = self._voice_path(voice_id)
        if not vp.exists():
            return False
        shutil.rmtree(vp)
        return True

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
            "voices_count": len(list(self._voices_dir.iterdir())),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_xtts_adapter.py -v`
Expected: All tests pass (mocked imports, no GPU needed)

- [ ] **Step 5 (VRAM): Register GPU resource with VRAMScheduler**

In the XTTSAdapter tests, add the GPU registration test to Step 1 if not already present.
Then in the actual adapter, at the end of `__init__()`, add:

```python
from runtime.gpu.vram_scheduler import vram_scheduler, GPUResource, VRAMProfile

vram_scheduler.update_gpu(GPUResource(
    gpu_id="xtts-v2",
    node_id="localhost",
    worker_port=0,
    vram=VRAMProfile(total_mb=5120, used_mb=0, free_mb=5120),
    model_loaded="xtts-v2",
    queue_depth=0,
    healthy=True,
    tier=0,
    metadata={"type": "tts", "provider": "xtts"},
))
```

This makes TTS visible in GPU health metrics and the dashboard.

- [ ] **Step 6: Commit**

```bash
git add providers/xtts_adapter.py tests/test_xtts_adapter.py
git commit -m "feat(tts): add XTTSAdapter with voice cloning support"
```

---

### Task 4: Audio Router — API Endpoints

**Files:**
- Create: `router/audio_router.py`
- Create: `tests/test_audio_router.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_audio_router.py`:

```python
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app as fastapi_app

AUDIO_MODULE = "router.audio_router"
ROUTER_MODULE = "runtime.orchestration.provider_router"


@pytest.fixture
def mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.tts.return_value = b"fake-wav-bytes"
    adapter.list_voices.return_value = [
        {"voice_id": "v1", "name": "Voice 1", "language": "en",
         "created_at": "2026-01-01T00:00:00", "duration_seconds": 5.0},
    ]
    adapter.register_voice.return_value = {
        "voice_id": "new-id", "name": "New Voice", "language": "en",
        "created_at": "2026-06-29T00:00:00", "duration_seconds": 3.0,
    }
    adapter.delete_voice.return_value = True
    return adapter


@pytest.fixture
def client(mock_adapter: MagicMock) -> TestClient:
    with patch(f"{ROUTER_MODULE}.adapter", return_value=mock_adapter):
        with patch(f"{AUDIO_MODULE}.settings.tts_enabled", True):
            # Ensure router is included
            from router.audio_router import router
            if router not in fastapi_app.routes:
                fastapi_app.include_router(router)
            yield TestClient(fastapi_app)


class TestAudioSpeech:
    def test_tts_success(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "Hello world",
            "voice": "v1",
        })
        assert resp.status_code == 200
        assert resp.content == b"fake-wav-bytes"
        assert "audio/wav" in resp.headers.get("content-type", "")

    def test_tts_requires_input(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={"voice": "v1"})
        assert resp.status_code == 422

    def test_tts_requires_voice(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={"input": "hello"})
        assert resp.status_code == 422

    def test_tts_with_format(self, client: TestClient, mock_adapter) -> None:
        mock_adapter.tts.return_value = b"fake-wav-bytes"
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "Hi",
            "voice": "v1",
            "response_format": "wav",
        })
        assert resp.status_code == 200

    def test_tts_with_language(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "您好",
            "voice": "v1",
            "language": "zh-tw",
        })
        assert resp.status_code == 200

    def test_tts_returns_404_for_unknown_voice(self, client: TestClient, mock_adapter) -> None:
        from providers.tts_base import TTSProviderError
        mock_adapter.tts.side_effect = TTSProviderError("voice not found", status_code=404)
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "hello",
            "voice": "nonexistent",
        })
        assert resp.status_code == 404


class TestVoicesAPI:
    def test_list_voices(self, client: TestClient) -> None:
        resp = client.get("/v1/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Voice 1"

    def test_register_voice(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/voices",
            data={"name": "New Voice", "language": "en"},
            files={"file": ("ref.wav", b"fake-audio-data", "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "New Voice"

    def test_register_voice_missing_name(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/voices",
            files={"file": ("ref.wav", b"data", "audio/wav")},
        )
        assert resp.status_code == 422

    def test_delete_voice(self, client: TestClient) -> None:
        resp = client.delete("/v1/voices/v1")
        assert resp.status_code == 204

    def test_delete_voice_not_found(self, client: TestClient, mock_adapter) -> None:
        mock_adapter.delete_voice.return_value = False
        resp = client.delete("/v1/voices/nonexistent")
        assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_router.py -v`
Expected: ImportError for `router.audio_router`

- [ ] **Step 3: Write the audio router**

Create `router/audio_router.py`:

```python
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from config.settings import settings
from providers.tts_base import TTSProviderError
from runtime.orchestration.provider_router import adapter as get_adapter

router = APIRouter(tags=["audio"])

AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


def _resolve_adapter():
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is not enabled")
    return get_adapter("xtts")


@router.post("/v1/audio/speech")
async def create_speech(payload: dict[str, Any]) -> Response:
    model = payload.get("model", "xtts-v2")
    text = payload.get("input", "")
    voice = payload.get("voice", "")
    response_format = payload.get("response_format", "wav")
    language = payload.get("language", "")
    speed = payload.get("speed", 1.0)

    if not text:
        raise HTTPException(status_code=422, detail="input is required")
    if not voice:
        raise HTTPException(status_code=422, detail="voice is required")

    fmt = response_format.lower()
    content_type = AUDIO_CONTENT_TYPES.get(fmt, "audio/wav")

    adapter = _resolve_adapter()
    try:
        audio_bytes = adapter.tts({
            "model": model,
            "input": text,
            "voice": voice,
            "language": language,
            "speed": speed,
        })
    except TTSProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    if fmt != "wav":
        audio_bytes = _convert_format(audio_bytes, fmt)

    return Response(content=audio_bytes, media_type=content_type)


def _convert_format(wav_bytes: bytes, target_format: str) -> bytes:
    import subprocess
    fmt_map = {
        "mp3": ["-f", "mp3", "-b:a", "192k"],
        "opus": ["-f", "opus", "-b:a", "96k"],
        "flac": ["-f", "flac"],
    }
    args = fmt_map.get(target_format)
    if args is None:
        return wav_bytes
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", *args, "pipe:1"],
            input=wav_bytes, capture_output=True, check=True, timeout=120,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return wav_bytes


@router.get("/v1/voices")
async def list_voices() -> list[dict[str, Any]]:
    adapter = _resolve_adapter()
    return adapter.list_voices()


@router.post("/v1/voices", status_code=200)
async def register_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form(default=""),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_adapter()
    return adapter.register_voice(
        name=name,
        audio_data=audio_data,
        language=language,
        content_type=file.content_type or "audio/wav",
    )


@router.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    adapter = _resolve_adapter()
    deleted = adapter.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    return Response(status_code=204)
```

- [ ] **Step 4: Register the router in the main FastAPI app**

The main app entry point (likely `main.py` or `router/__init__.py`) includes routers.
Find where other routers (like `router/openai_router.py`) are included and add:

```python
if settings.tts_enabled:
    from router.audio_router import router as audio_router
    app.include_router(audio_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_audio_router.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add router/audio_router.py tests/test_audio_router.py
git commit -m "feat(tts): add /v1/audio/speech and /v1/voices/* API endpoints"
```

---

### Task 5: Provider Router Integration

**Files:**
- Modify: `runtime/orchestration/provider_router.py`
- Modify: the file that includes routers into the FastAPI app (check `main.py` or `app/__init__.py`)

- [ ] **Step 1: Add xtts to adapter() in provider_router.py**

In `runtime/orchestration/provider_router.py`, add `"xtts"` handling in the `adapter()` function (or a similar factory):

Near the `_CLOUD_ADAPTERS` dict or adapter factory, add:

```python
from providers.xtts_adapter import XTTSAdapter
from config.settings import settings

_tts_adapter: XTTSAdapter | None = None


def _get_tts_adapter() -> XTTSAdapter:
    global _tts_adapter
    if _tts_adapter is None and settings.tts_enabled:
        _tts_adapter = XTTSAdapter(
            model_name=settings.tts_model_name,
            device=settings.tts_device,
            voices_dir=settings.tts_voices_dir,
            models_dir=settings.tts_models_dir,
        )
    return _tts_adapter
```

Then in the existing `adapter()` function, add the xtts branch:

```python
if provider == "xtts":
    return _get_tts_adapter()
```

Also add `"xtts/"` to `ROUTE_PREFIXES`:

```python
ROUTE_PREFIXES = {
    # ... existing entries ...
    "xtts/": "xtts",
}
```

- [ ] **Step 2: Wire the audio_router into the FastAPI app**

Find where routers are included (likely `main.py` or a similar app factory). Add:

```python
if settings.tts_enabled:
    from router.audio_router import router as audio_router
    app.include_router(audio_router)
```

- [ ] **Step 3: Wire /v1/models integration (verify xtts-v2 appears in model list)**

Since `xtts-v2` was added to `config/models.yaml` in Task 1, and the existing models
endpoint reads from the model registry, it should appear automatically. Verify with:

```bash
curl -s http://localhost:8000/v1/models | python -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d.get('data',[]) if 'xtts' in m['id']])"
```

If xtts-v2 does NOT appear, add it to the model list in `router/openai/models_adapter.py`
or wherever the `/v1/models` response is assembled.

- [ ] **Step 4: Verify full integration test**

Run all tests to make sure nothing is broken:

```bash
pytest tests/ -x -v --timeout=30 2>&1 | head -80
```

Expected: Previous 296+ tests still passing, plus ~30 new TTS tests.

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/provider_router.py
git commit -m "feat(tts): wire xtts provider into router adapter + register audio_router"
```

---

### Task 6: README & Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `TASK.md`

- [ ] **Step 1: Update README.md**

Add a TTS row to the Provider Adapters table and a Local TTS section:

```markdown
### Local TTS (Text-to-Speech)

XTTS-v2 provides local GPU-accelerated TTS with voice cloning.

| Provider | File | Capabilities |
|----------|------|-------------|
| XTTS-v2 | `providers/xtts_adapter.py` | audio |

**Configuration:**
- `AIIH_TTS_ENABLED=true` — enable TTS
- `AIIH_TTS_DEVICE=cuda:0` — GPU device
- `AIIH_TTS_VOICES_DIR=data/voices` — cloned voice storage

**API endpoints:**
- `POST /v1/audio/speech` — generate speech (OpenAI-compatible)
- `POST /v1/voices` — register a voice
- `GET /v1/voices` — list voices
- `DELETE /v1/voices/{voice_id}` — delete a voice

**Installation:**
```bash
pip install -e ".[tts]"
```

**Example:**
```bash
# Register a voice
curl -X POST http://localhost:8000/v1/voices \
  -F "name=fred-voice" -F "file=@sample.wav"

# Generate speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"xtts-v2","input":"Hello world","voice":"<voice_id>"}' \
  --output hello.wav
```

- [ ] **Step 2: Update TASK.md**

Append a progress row:

```markdown
| 2026-06-29 | Phase 24 | Local XTTS-v2 TTS Provider: TTSProviderAdapter ABC, XTTSAdapter (GPU), /v1/audio/speech + /v1/voices/* API, voice cloning, ffmpeg format conversion |
```

- [ ] **Step 3: Commit**

```bash
git add README.md TASK.md
git commit -m "docs(tts): update README and TASK.md with XTTS-v2 TTS provider"
```
