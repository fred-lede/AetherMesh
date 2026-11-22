from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
from typing import Any

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.mlx_whisper_adapter")

_MLX_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

_FP16_VALUES = {"float16", "fp16", "half"}


def _resolve_repo(model_name: str) -> str:
    name = str(model_name or "").strip()
    if not name:
        return _MLX_REPOS["large-v3"]
    if "/" in name:
        return name
    return _MLX_REPOS.get(name.lower(), name)


class _MlxSegment:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


class _MlxInfo:
    def __init__(self, language: str) -> None:
        self.language = language


class _MlxTranscriber:
    """faster-whisper shaped entry point the realtime pipeline can stream with.

    `transcribe(samples, ...)` returns `(segments, info)` and runs on the worker
    thread, so MLX never touches an array created elsewhere.
    """

    def __init__(self, repo: str, fp16: bool) -> None:
        self._repo = repo
        self._fp16 = fp16

    def transcribe(self, samples: Any, language: str = "", **kwargs: Any) -> tuple[list[Any], Any]:
        result = _get_worker().run(
            {
                "repo": self._repo,
                "fp16": self._fp16,
                "audio": samples,
                "task": kwargs.get("task", "transcribe"),
                "temperature": kwargs.get("temperature", 0.0),
                "language": language,
                "prompt": kwargs.get("prompt", ""),
            }
        )
        segments = [
            _MlxSegment(str(seg.get("text") or ""), float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
            for seg in result.get("segments") or []
        ]
        return segments, _MlxInfo(result.get("language") or language)


class _MlxWorker:
    """Runs every MLX call on one thread.

    MLX arrays are only valid on the thread that created them, so a model
    loaded on the preload thread cannot be used from a request thread.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mlx-whisper")
        self._thread.start()

    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"job": job, "done": threading.Event(), "response": None}
        self._jobs.put(entry)
        entry["done"].wait()
        response = entry["response"]
        if isinstance(response, Exception):
            raise response
        return response

    def _loop(self) -> None:
        while True:
            entry = self._jobs.get()
            try:
                entry["response"] = self._execute(entry["job"])
            except Exception as exc:
                entry["response"] = exc
            finally:
                entry["done"].set()

    @staticmethod
    def _execute(job: dict[str, Any]) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_whisper import transcribe
        from mlx_whisper.transcribe import ModelHolder

        dtype = mx.float16 if job["fp16"] else mx.float32
        model = ModelHolder.get_model(job["repo"], dtype)
        if job.get("load_only"):
            return {"model_loaded": model is not None}

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": job["repo"],
            "task": job["task"],
            "temperature": job["temperature"],
            "fp16": job["fp16"],
        }
        if job.get("language"):
            kwargs["language"] = job["language"]
        if job.get("prompt"):
            kwargs["initial_prompt"] = job["prompt"]

        result = transcribe(job["audio"], **kwargs)
        return {
            "text": str(result.get("text") or "").strip(),
            "segments": list(result.get("segments") or []),
            "language": str(result.get("language") or ""),
        }


_worker: _MlxWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _MlxWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _MlxWorker()
        return _worker


class MLXWhisperAdapter(ASRProviderAdapter):
    provider_name = "mlx_whisper"

    def __init__(self, model_name: str = "large-v3", compute_type: str = "float16") -> None:
        self._model_name = model_name
        self._repo = _resolve_repo(model_name)
        self._fp16 = str(compute_type or "").strip().lower() in _FP16_VALUES
        logger.info("MLX Whisper loading repo=%s dtype=%s", self._repo, "float16" if self._fp16 else "float32")
        self._model = _MlxTranscriber(self._repo, self._fp16)
        self._model_loaded = bool(_get_worker().run({"repo": self._repo, "fp16": self._fp16, "load_only": True}).get("model_loaded"))

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

            result = _get_worker().run(
                {
                    "repo": self._repo,
                    "fp16": self._fp16,
                    "audio": tmp.name,
                    "task": task,
                    "temperature": temperature,
                    "language": language,
                    "prompt": prompt,
                }
            )
            return {"text": result["text"]}
        except ValueError as exc:
            raise ASRProviderError(str(exc), status_code=400) from exc
        except Exception as exc:
            raise ASRProviderError(f"ASR transcription failed: {exc}", status_code=503) from exc
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model_loaded,
            "model": self._model_name,
            "device": "mps",
        }
