from __future__ import annotations

import io
import logging
import wave
from typing import Any

import numpy as np

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.qwen_asr_adapter")

_LANGUAGE_CODES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
}


class _QwenSegment:
    def __init__(self, text: str, language: str = "") -> None:
        self.text = text
        self.language = language


class _QwenInfo:
    def __init__(self, language: str = "") -> None:
        self.language = language


class _QwenTranscriber:
    def __init__(self, model_name: str, device: str, device_index: int, download_dir: str | None) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRProviderError(
                "qwen-asr is not installed; run: pip install -U qwen-asr",
                status_code=503,
            ) from exc
        normalized = device.strip().lower()
        if normalized == "cuda":
            dtype, device_map = torch.bfloat16, f"cuda:{device_index}"
        elif normalized == "mps":
            dtype, device_map = torch.float32, "mps"
        else:
            dtype, device_map = torch.float32, "cpu"
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
        }
        if download_dir:
            kwargs["cache_dir"] = download_dir
        self._model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)

    def transcribe(self, samples: np.ndarray, language: str = "") -> tuple[list[Any], Any]:
        qwen_language = _LANGUAGE_CODES.get(language.strip().lower()) if language else None
        results = self._model.transcribe(
            audio=(np.asarray(samples, dtype=np.float32), 16000),
            language=qwen_language,
            return_time_stamps=False,
        )
        first = results[0]
        text = str(getattr(first, "text", "") or "")
        detected = str(getattr(first, "language", "") or "")
        return [_QwenSegment(text, detected)], _QwenInfo(detected)


class QwenASRAdapter(ASRProviderAdapter):
    provider_name = "qwen_asr"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        device_index: int = 0,
        download_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._device_index = device_index
        self._download_dir = download_dir
        self._model = self._load_model()

    def _load_model(self) -> _QwenTranscriber:
        return _QwenTranscriber(self._model_name, self._device, self._device_index, self._download_dir)

    @staticmethod
    def _decode_wav(audio: bytes) -> np.ndarray:
        if len(audio) < 12 or audio[:4] != b"RIFF":
            raise ASRProviderError("qwen_asr file input supports 16kHz WAV only", status_code=400)
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except wave.Error as exc:
            raise ASRProviderError("qwen_asr file input supports 16kHz WAV only", status_code=400) from exc
        if sample_rate != 16000:
            raise ASRProviderError(f"qwen_asr file input must be 16kHz WAV (got {sample_rate}Hz)", status_code=400)
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ASRProviderError(f"unsupported WAV sample width {sampwidth}", status_code=400)
        samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if sampwidth == 1:
            samples = (samples - 128.0) / 128.0
        elif sampwidth == 2:
            samples /= 32768.0
        elif sampwidth == 4:
            samples /= 2147483648.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples

    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        try:
            samples = self._decode_wav(audio)
            segments, _info = self._model.transcribe(samples, language=language)
            return {"text": segments[0].text}
        except ASRProviderError:
            raise
        except Exception as exc:
            raise ASRProviderError(f"ASR transcription failed: {exc}", status_code=503) from exc

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
            "device_index": self._device_index,
        }
