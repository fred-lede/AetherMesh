from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.qwen_asr_adapter")

_WORKER_PATH = Path(__file__).resolve().parents[1] / "runtime" / "asr" / "qwen_asr_worker.py"

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
        normalized = device.strip().lower()
        if normalized == "cuda":
            dtype, device_map = "bfloat16", f"cuda:{device_index}"
        elif normalized == "mps":
            dtype, device_map = "float32", "mps"
        else:
            dtype, device_map = "float32", "cpu"

        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        python = os.getenv("AIIH_QWEN_ASR_PYTHON", "").strip()
        if python:
            self._proc = subprocess.Popen(
                [python, str(_WORKER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            response = self._exchange(
                {
                    "model": model_name,
                    "dtype": dtype,
                    "device_map": device_map,
                    "cache_dir": download_dir,
                }
            )
            if not response.get("ok"):
                self.close()
                raise ASRProviderError(
                    str(response.get("error") or "qwen-asr worker failed to start"),
                    status_code=503,
                )
            logger.info("qwen-asr worker started via %s on %s", python, device_map)
            return

        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRProviderError(
                "qwen-asr dependencies failed to import: %s. "
                "Run: pip install -U qwen-asr. "
                "If torch is the failing import, check for conflicting NCCL/CUDA "
                "wheel versions (e.g. both nvidia-nccl-cu12 and -cu13 installed)."
                % exc,
                status_code=503,
            ) from exc
        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if download_dir:
            kwargs["cache_dir"] = download_dir
        self._model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise ASRProviderError("qwen-asr worker is not running", status_code=503)
        with self._lock:
            proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            proc.stdin.flush()
            line = proc.stdout.readline()
        if not line:
            raise ASRProviderError("qwen-asr worker exited unexpectedly", status_code=503)
        return json.loads(line)

    def transcribe(self, samples: np.ndarray, language: str = "", **kwargs) -> tuple[list[Any], Any]:
        qwen_language = _LANGUAGE_CODES.get(language.strip().lower()) if language else None
        if self._proc is not None:
            payload = {
                "samples": base64.b64encode(np.asarray(samples, dtype=np.float32).tobytes()).decode("ascii"),
                "language": qwen_language,
            }
            response = self._exchange(payload)
            if not response.get("ok"):
                raise ASRProviderError(
                    str(response.get("error") or "qwen-asr transcription failed"),
                    status_code=503,
                )
            segments = response.get("segments") or []
            first = segments[0] if segments else {}
            text = str(first.get("text", "") or "")
            detected = str(first.get("language", "") or "")
            return [_QwenSegment(text, detected)], _QwenInfo(detected)

        results = self._model.transcribe(
            audio=(np.asarray(samples, dtype=np.float32), 16000),
            language=qwen_language,
            return_time_stamps=False,
        )
        first = results[0]
        text = str(getattr(first, "text", "") or "")
        detected = str(getattr(first, "language", "") or "")
        return [_QwenSegment(text, detected)], _QwenInfo(detected)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        self._proc = None
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __del__(self) -> None:
        self.close()


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

    def close(self) -> None:
        close = getattr(getattr(self, "_model", None), "close", None)
        if callable(close):
            close()
        self._model = None

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
