from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from providers.tts_base import TTSProviderError


def _fake_load_success(self):
    model = MagicMock()
    mock_output = MagicMock()
    mock_output.get.side_effect = lambda k, d=None: {"tts_speech": MagicMock()}.get(k, d)
    model.inference_sft.return_value = [mock_output]
    model.inference_zero_shot.return_value = [mock_output]
    self._model = model
    self._load_error = None


@pytest.fixture
def mock_cosyvoice():
    from providers.cosyvoice_adapter import CosyVoiceAdapter
    with patch.object(CosyVoiceAdapter, "_load_model", _fake_load_success):
        yield


@pytest.fixture
def adapter(tmp_path, mock_cosyvoice):
    from providers.cosyvoice_adapter import CosyVoiceAdapter
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    a = CosyVoiceAdapter(
        model_name="test-model",
        device="cpu",
        voices_dir=str(voices_dir),
    )
    a._voices_dir = voices_dir
    return a


class TestCosyVoiceInit:
    def test_loads_model_on_init(self, tmp_path):
        from providers.cosyvoice_adapter import CosyVoiceAdapter
        with patch.object(CosyVoiceAdapter, "_load_model", _fake_load_success):
            a = CosyVoiceAdapter(model_name="test-model", device="cpu", voices_dir=str(tmp_path / "v"))
            assert a._model is not None
            assert a._load_error is None

    def test_handles_load_failure(self, tmp_path):
        def _fake_fail(self):
            self._model = None
            self._load_error = "mock fail"

        from providers.cosyvoice_adapter import CosyVoiceAdapter
        with patch.object(CosyVoiceAdapter, "_load_model", _fake_fail):
            a = CosyVoiceAdapter(model_name="test", device="cpu", voices_dir=str(tmp_path / "v"))
            assert a._model is None
            assert "mock fail" in (a._load_error or "")

    def test_creates_voices_dir(self, tmp_path, mock_cosyvoice):
        voices_dir = tmp_path / "nonexistent-voices"
        assert not voices_dir.exists()
        from providers.cosyvoice_adapter import CosyVoiceAdapter
        CosyVoiceAdapter(model_name="test-model", device="cpu", voices_dir=str(voices_dir))
        assert voices_dir.exists()


class TestCosyVoiceTTS:
    def test_returns_wav_bytes(self, adapter):
        result = adapter.tts({"input": "Hello", "voice": "", "language": "en", "speed": 1.0})
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_raises_on_no_output(self, adapter):
        adapter._model.inference_sft.return_value = []
        adapter._model.inference_zero_shot.return_value = []
        with pytest.raises(TTSProviderError, match="no audio"):
            adapter.tts({"input": "Hi", "voice": "", "language": "en", "speed": 1.0})

    def test_uses_sft_on_missing_voice_dir(self, adapter):
        mock_model = adapter._model
        mock_output = MagicMock()
        mock_output.get.side_effect = lambda k, d=None: {"tts_speech": MagicMock()}.get(k, d)
        mock_model.inference_sft.return_value = [mock_output]

        adapter.tts({"input": "Hello", "voice": "nonexistent", "language": "en", "speed": 1.0})
        mock_model.inference_sft.assert_called_once_with("Hello", spk_id="nonexistent")

    def test_uses_zero_shot_with_registered_voice(self, adapter):
        mock_model = adapter._model
        mock_output = MagicMock()
        mock_output.get.side_effect = lambda k, d=None: {"tts_speech": MagicMock()}.get(k, d)
        mock_model.inference_zero_shot.return_value = [mock_output]

        vp = adapter._voices_dir / "test-voice"
        vp.mkdir()
        (vp / "reference.wav").write_bytes(b"\x00" * 1000)
        (vp / "reference.txt").write_text("prompt text", encoding="utf-8")
        (vp / "meta.json").write_text(json.dumps({"voice_id": "test-voice", "name": "Test"}))

        with patch.object(adapter, "_load_audio", return_value=MagicMock()):
            adapter.tts({"input": "Hello world", "voice": "test-voice", "language": "en", "speed": 1.0})
        mock_model.inference_zero_shot.assert_called_once()
        args, _ = mock_model.inference_zero_shot.call_args
        assert args[0] == "Hello world"

    def test_applies_speed(self, adapter):
        original = b"\x00" * 100
        result = adapter._apply_speed(original, 1.5)
        assert result == original


class TestCosyVoiceVoices:
    def test_list_voices_empty(self, adapter):
        assert adapter.list_voices() == []

    def test_list_voices_with_data(self, adapter):
        for v_id in ("v1", "v2"):
            vp = adapter._voices_dir / v_id
            vp.mkdir()
            (vp / "meta.json").write_text(json.dumps({"voice_id": v_id, "name": f"Voice {v_id}"}))
        voices = adapter.list_voices()
        assert len(voices) == 2

    def test_register_voice_creates_meta(self, adapter):
        result = adapter.register_voice(name="Test Voice", audio_data=b"\x00" * 1000, language="en")
        assert "voice_id" in result
        assert result["name"] == "Test Voice"
        vp = adapter._voices_dir / result["voice_id"]
        assert (vp / "reference.wav").exists()
        assert (vp / "reference.txt").exists()
        assert (vp / "meta.json").exists()

    def test_delete_voice(self, adapter):
        vp = adapter._voices_dir / "to-delete"
        vp.mkdir()
        (vp / "meta.json").write_text("{}")
        assert adapter.delete_voice("to-delete") is True
        assert not vp.exists()

    def test_delete_voice_not_found(self, adapter):
        assert adapter.delete_voice("nonexistent") is False


class TestCosyVoiceHealth:
    def test_health_check(self, adapter):
        hc = adapter.health_check()
        assert hc["provider"] == "cosyvoice"
        assert hc["model_loaded"] is True
        assert hc["device"] == "cpu"
        assert hc["voices_count"] == 0


class TestCosyVoiceAudio:
    def test_get_audio_duration_returns_zero_on_error(self, adapter):
        duration = adapter._get_audio_duration(b"\x00" * 100)
        assert duration == 0.0

    def test_cat_audio_empty_raises(self, adapter):
        with pytest.raises(TTSProviderError, match="no audio"):
            adapter._cat_audio([])
