from __future__ import annotations

import os
import tempfile
from typing import Any

from providers.asr_base import ASRProviderAdapter, ASRProviderError


class FasterWhisperAdapter(ASRProviderAdapter):
    provider_name = "faster_whisper"

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        download_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._download_dir = download_dir
        self._model = self._load_model()

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel
        kwargs: dict[str, Any] = {
            "device": self._device,
            "compute_type": self._compute_type,
        }
        if self._download_dir:
            kwargs["download_root"] = os.path.abspath(self._download_dir)
        return WhisperModel(self._model_name, **kwargs)

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
            raise ASRProviderError(f"ASR transcription failed: {e}", status_code=503) from e
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
