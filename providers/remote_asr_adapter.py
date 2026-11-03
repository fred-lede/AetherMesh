from __future__ import annotations

import io
import logging
import os
import wave
from typing import Any

import numpy as np
import requests

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.remote_asr_adapter")


def pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buf.getvalue()


class _RemoteSegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.language = ""


class _RemoteInfo:
    language = ""


class _RemoteTranscriber:
    def __init__(self, node_id: str, base_url: str, token: str | None, timeout_s: float) -> None:
        self._node_id = node_id
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _fail(self, message: str, status_code: int = 503) -> ASRProviderError:
        return ASRProviderError(f"asr worker node={self._node_id} at {self._base_url} {message}", status_code=status_code)

    def transcribe_file(self, audio: bytes, language: str = "") -> str:
        files = {"file": ("upload.bin", audio, "application/octet-stream")}
        data = {"language": language} if language else {}
        return self._post(files, data)

    def transcribe(self, samples: np.ndarray, language: str = "", **kwargs) -> tuple[list[Any], Any]:
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2").tobytes()
        data = {"language": language} if language else {}
        text = self._post({"file": ("audio.wav", pcm16_to_wav(pcm16), "audio/wav")}, data)
        return [_RemoteSegment(text)], _RemoteInfo()

    def _post(self, files: dict[str, Any], data: dict[str, str]) -> str:
        try:
            response = requests.post(
                f"{self._base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=self._headers(),
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            raise self._fail(f"unreachable: {exc}") from exc
        if not response.ok:
            status_code = response.status_code if 400 <= response.status_code < 500 else 503
            raise self._fail(f"returned HTTP {response.status_code}: {response.text[:200]}", status_code=status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise self._fail("returned a non-JSON response") from exc
        return str(payload.get("text") or "")

    def health_check(self) -> dict[str, Any]:
        base = {"reachable": False, "node_id": self._node_id, "base_url": self._base_url}
        try:
            response = requests.get(f"{self._base_url}/health", headers=self._headers(), timeout=3)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return base
        return {
            "reachable": True,
            "node_id": self._node_id,
            "base_url": self._base_url,
            "engine": payload.get("engine"),
            "model": payload.get("model"),
            "model_loaded": payload.get("model_loaded"),
        }


class RemoteASRAdapter(ASRProviderAdapter):
    provider_name = "remote_asr"

    def __init__(self, node_id: str, base_url: str, timeout_s: float = 10.0, token: str | None = None) -> None:
        self._node_id = node_id
        self._base_url = base_url.rstrip("/")
        if token is None:
            token = os.getenv("AIIH_ASR_WORKER_TOKEN") or None
        self._model = _RemoteTranscriber(node_id, base_url, token=token, timeout_s=timeout_s)

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
            text = self._model.transcribe_file(audio, language=language)
            return {"text": text}
        except ASRProviderError:
            raise
        except Exception as exc:
            raise ASRProviderError(f"ASR transcription failed: {exc}", status_code=503) from exc

    def health_check(self) -> dict[str, Any]:
        return self._model.health_check()
