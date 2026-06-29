from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from providers.asr_base import ASRProviderError


@pytest.fixture(autouse=True)
def _mock_asr_deps() -> MagicMock:
    """Mock faster_whisper, soundfile, torch to prevent GPU imports."""
    from unittest.mock import patch
    with patch.dict("sys.modules"):
        import sys

        fw = MagicMock()
        fw.WhisperModel = MagicMock()
        fake_model = MagicMock()
        fake_segment = MagicMock()
        fake_segment.text = "hello world"
        fake_segment.start = 0.0
        fake_segment.end = 2.5
        fake_model.transcribe.return_value = [fake_segment], None
        fw.WhisperModel.return_value = fake_model
        sys.modules["faster_whisper"] = fw

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        sys.modules["soundfile"] = mock_sf

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
