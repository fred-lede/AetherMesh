from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.tts_base import TTSProviderError


@pytest.fixture(autouse=True)
def _mock_tts_deps() -> MagicMock:
    """Mock TTS, torch, and soundfile to prevent GPU/PyTorch imports."""
    with patch.dict("sys.modules"):
        fake_tts = MagicMock()
        fake_tts.tts.return_value = np.zeros((24000,), dtype=np.float32)
        fake_tts.to.return_value = None
        fake_tts.synthesizer.tts_model.get_conditioning_latents.return_value = (
            np.zeros((1, 1024), dtype=np.float32),
            np.zeros((1, 256), dtype=np.float32),
        )
        fake_tts.synthesizer.tts_model.inference.return_value = {
            "wav": np.zeros((24000,), dtype=np.float32),
        }

        tts_api = MagicMock()
        tts_api.TTS.return_value = fake_tts

        import sys
        sys.modules["TTS"] = tts_api
        sys.modules["TTS.api"] = tts_api

        mock_sf = MagicMock()
        mock_sf.info.return_value.duration = 3.0
        sys.modules["soundfile"] = mock_sf

        sys.modules["torch"] = MagicMock()
        sys.modules["transformers"] = MagicMock()
        sys.modules["transformers.pytorch_utils"] = MagicMock()

        yield fake_tts


@pytest.fixture
def voices_dir() -> Path:
    d = Path(tempfile.mkdtemp()) / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def adapter(_mock_tts_deps: MagicMock, voices_dir: Path) -> MagicMock:
    from providers.xtts_adapter import XTTSAdapter
    return XTTSAdapter(
        model_name="test-model",
        device="cpu",
        voices_dir=str(voices_dir),
    )


class TestXTTSAdapterInit:
    def test_provider_name(self, adapter: MagicMock) -> None:
        assert adapter.provider_name == "xtts"

    def test_voices_dir_created(self, adapter: MagicMock, voices_dir: Path) -> None:
        assert voices_dir.is_dir()

    def test_model_loaded(self, adapter: MagicMock) -> None:
        assert adapter._model is not None


class TestXTTSAdapterTTS:
    def test_tts_returns_bytes(self, adapter: MagicMock) -> None:
        adapter._load_embedding = MagicMock(  # type: ignore[method-assign]
            return_value=(np.zeros((1, 1024)), np.zeros((1, 256)))
        )
        result = adapter.tts({
            "voice": "test-id",
            "input": "Hello world",
            "language": "en",
        })
        assert isinstance(result, bytes)

    def test_tts_missing_voice_raises(self, adapter: MagicMock) -> None:
        adapter._load_embedding = MagicMock(  # type: ignore[method-assign]
            side_effect=TTSProviderError("not found", status_code=404)
        )
        with pytest.raises(TTSProviderError) as exc:
            adapter.tts({"voice": "nonexistent", "input": "hello"})
        assert exc.value.status_code == 404

    def test_tts_applies_speed(self, adapter: MagicMock) -> None:
        applied: list[float] = []

        def fake_speed(data: bytes, speed: float) -> bytes:
            applied.append(speed)
            return data

        adapter._apply_speed = fake_speed  # type: ignore[method-assign]
        adapter._load_embedding = MagicMock(  # type: ignore[method-assign]
            return_value=(np.zeros((1, 1024)), np.zeros((1, 256)))
        )
        adapter.tts({"voice": "x", "input": "hi", "speed": 1.5})
        assert applied == [1.5]


class TestXTTSAdapterVoiceCRUD:
    def test_register_and_list(self, adapter: MagicMock) -> None:
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

    def test_delete_voice(self, adapter: MagicMock) -> None:
        meta = adapter.register_voice(name="del-me", audio_data=b"data")
        assert adapter.delete_voice(meta["voice_id"]) is True
        assert adapter.delete_voice("nonexistent") is False

    def test_health_check(self, adapter: MagicMock) -> None:
        health = adapter.health_check()
        assert health["provider"] == "xtts"
        assert "model_loaded" in health
        assert "device" in health
        assert "voices_count" in health
