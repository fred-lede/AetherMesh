from __future__ import annotations

import io
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import providers.qwen_asr_adapter as qwen_module
from providers.asr_base import ASRProviderError
from providers.qwen_asr_adapter import QwenASRAdapter


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _wav_bytes_8bit(raw: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    return buf.getvalue()


@pytest.fixture
def fake_qwen(monkeypatch):
    calls: list[dict] = []
    instance = SimpleNamespace(init_path=None, init_kwargs=None)
    instance.transcribe = lambda audio, language=None, return_time_stamps=False: (
        calls.append({"audio": audio, "language": language}) or [SimpleNamespace(text="你好世界", language="Chinese")]
    )
    fake = types.ModuleType("qwen_asr")

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            instance.init_path = path
            instance.init_kwargs = kwargs
            return instance

    fake.Qwen3ASRModel = FakeQwen3ASRModel
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "qwen_asr", fake)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return instance, calls


def _samples() -> np.ndarray:
    return np.full(16000, 0.01, dtype=np.float32)


def test_stream_contract_shape(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(model_name="Qwen/Qwen3-ASR-1.7B", device="cpu")
    segments, info = adapter._model.transcribe(_samples(), language="")
    assert segments[0].text == "你好世界"
    assert info.language == "Chinese"


def test_stream_transcribe_tolerates_whisper_convention_kwargs(fake_qwen):
    instance, _ = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    segments, _info = adapter._model.transcribe(_samples(), language="", task="transcribe")
    assert segments[0].text == "你好世界"


def test_language_auto_detect_passes_none(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    adapter._model.transcribe(_samples(), language="")
    assert calls[0]["language"] is None


@pytest.mark.parametrize("code,expected", [("zh", "Chinese"), ("en", "English"), ("ja", "Japanese"), ("klingon", None)])
def test_language_code_mapping(fake_qwen, code, expected):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    adapter._model.transcribe(_samples(), language=code)
    assert calls[-1]["language"] == expected


def test_from_pretrained_device_and_cache(fake_qwen):
    instance, _ = fake_qwen
    QwenASRAdapter(model_name="Qwen/Qwen3-ASR-0.6B", device="cuda", device_index=1, download_dir="data/asr_models")
    assert instance.init_path == "Qwen/Qwen3-ASR-0.6B"
    assert instance.init_kwargs["device_map"] == "cuda:1"
    assert instance.init_kwargs["cache_dir"] == "data/asr_models"


def test_cpu_uses_float32(fake_qwen):
    instance, _ = fake_qwen
    QwenASRAdapter(device="cpu")
    assert instance.init_kwargs["device_map"] == "cpu"
    assert instance.init_kwargs["dtype"] == "float32"


def test_mps_uses_float32_and_mps_device_map(fake_qwen):
    instance, _ = fake_qwen
    QwenASRAdapter(device="mps", device_index=1)
    assert instance.init_kwargs["device_map"] == "mps"
    assert instance.init_kwargs["dtype"] == "float32"


@pytest.mark.parametrize("raw", ["MPS ", " mps"])
def test_mps_device_normalized(fake_qwen, raw):
    instance, _ = fake_qwen
    QwenASRAdapter(device=raw)
    assert instance.init_kwargs["device_map"] == "mps"


def test_bytes_transcribe_returns_text(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    result = adapter.transcribe(_wav_bytes(_samples()))
    assert result == {"text": "你好世界"}
    audio, sr = calls[0]["audio"]
    assert sr == 16000
    assert audio.dtype == np.float32


def test_bytes_rejects_non_wav(fake_qwen):
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"not a wav file at all")
    assert excinfo.value.status_code == 400


def test_bytes_rejects_wrong_sample_rate(fake_qwen):
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(_wav_bytes(_samples(), sample_rate=44100))
    assert excinfo.value.status_code == 400
    assert "16kHz" in str(excinfo.value)


def test_bytes_decodes_8bit_wav_as_unsigned(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    adapter.transcribe(_wav_bytes_8bit(bytes([0, 128, 255])))
    audio, sr = calls[0]["audio"]
    assert sr == 16000
    np.testing.assert_allclose(audio, [-1.0, 0.0, 0.9921875], atol=1e-6)


def test_bytes_rejects_short_riff_prefix(fake_qwen):
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"RIFF")
    assert excinfo.value.status_code == 400


def test_inference_error_wrapped_as_503(fake_qwen):
    instance, _ = fake_qwen
    instance.transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda exploded"))
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(_wav_bytes(_samples()))
    assert excinfo.value.status_code == 503


def test_missing_package_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    with pytest.raises(ASRProviderError) as excinfo:
        QwenASRAdapter(device="cpu")
    assert "pip install" in str(excinfo.value)
    assert excinfo.value.status_code == 503


def test_health_check_shape(fake_qwen):
    adapter = QwenASRAdapter(model_name="m", device="cuda", device_index=2)
    assert adapter.health_check() == {"provider": "qwen_asr", "model_loaded": True, "device": "cuda", "device_index": 2}
