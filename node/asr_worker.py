from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config.settings import settings

logger = logging.getLogger("node.asr_worker")

app = FastAPI(title="AetherMesh ASR Worker", version="5.0.0")

_adapter: Any = None
_adapter_lock = threading.Lock()


def _build_adapter() -> Any:
    if settings.asr_provider == "qwen_asr":
        from providers.qwen_asr_adapter import QwenASRAdapter

        return QwenASRAdapter(
            model_name=settings.asr_model if "/" in settings.asr_model else "Qwen/Qwen3-ASR-1.7B",
            device=settings.asr_device,
            device_index=settings.asr_device_index,
            download_dir=settings.asr_models_dir,
        )
    from providers.faster_whisper_adapter import FasterWhisperAdapter

    return FasterWhisperAdapter(
        model_name=settings.asr_model,
        device=settings.asr_device,
        device_index=settings.asr_device_index,
        compute_type=settings.asr_compute_type,
        download_dir=settings.asr_models_dir,
    )


def get_adapter() -> Any:
    global _adapter
    if _adapter is not None:
        return _adapter
    with _adapter_lock:
        if _adapter is None:
            logger.info(
                "ASR worker loading engine=%s model=%s device=%s index=%s",
                settings.asr_provider, settings.asr_model, settings.asr_device, settings.asr_device_index,
            )
            _adapter = _build_adapter()
            logger.info("ASR worker engine loaded")
    return _adapter


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "asr-worker",
        "engine": settings.asr_provider,
        "model": settings.asr_model,
        "model_loaded": _adapter is not None,
    }


@app.post("/v1/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), language: str = Form("")) -> dict[str, Any]:
    try:
        adapter = get_adapter()
    except Exception as exc:
        logger.exception("ASR worker adapter load failed")
        raise HTTPException(status_code=503, detail=f"ASR engine unavailable: {exc}") from exc
    audio = file.file.read()
    try:
        result = adapter.transcribe(audio, language=language)
    except Exception as exc:
        logger.error("ASR worker transcription failed: %s", exc)
        status = getattr(exc, "status_code", 503)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"text": str(result.get("text") or "")}


@app.on_event("startup")
def _startup_preload() -> None:
    if os.getenv("AIIH_ASR_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
        threading.Thread(target=get_adapter, daemon=True, name="asr-worker-preload").start()
