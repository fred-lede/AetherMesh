from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from config.settings import settings
from providers.tts_base import TTSProviderError
from runtime.orchestration.provider_router import adapter as get_adapter

router = APIRouter(tags=["audio"])

AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


def _resolve_adapter():
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is not enabled")
    return get_adapter("xtts")


@router.post("/v1/audio/speech")
async def create_speech(payload: dict[str, Any]) -> Response:
    text = payload.get("input", "")
    voice = payload.get("voice", "")
    response_format = payload.get("response_format", "wav")
    language = payload.get("language", "")
    speed = payload.get("speed", 1.0)

    if not text:
        raise HTTPException(status_code=422, detail="input is required")
    if not voice:
        raise HTTPException(status_code=422, detail="voice is required")

    fmt = response_format.lower()
    content_type = AUDIO_CONTENT_TYPES.get(fmt, "audio/wav")

    adapter = _resolve_adapter()
    try:
        audio_bytes = adapter.tts({
            "model": payload.get("model", "xtts-v2"),
            "input": text,
            "voice": voice,
            "language": language,
            "speed": speed,
        })
    except TTSProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    if fmt != "wav":
        audio_bytes = _convert_format(audio_bytes, fmt)

    return Response(content=audio_bytes, media_type=content_type)


def _convert_format(wav_bytes: bytes, target_format: str) -> bytes:
    fmt_map = {
        "mp3": ["-f", "mp3", "-b:a", "192k"],
        "opus": ["-f", "opus", "-b:a", "96k"],
        "flac": ["-f", "flac"],
    }
    args = fmt_map.get(target_format)
    if args is None:
        return wav_bytes
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", *args, "pipe:1"],
            input=wav_bytes, capture_output=True, check=True, timeout=120,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return wav_bytes


@router.get("/v1/voices")
async def list_voices() -> list[dict[str, Any]]:
    adapter = _resolve_adapter()
    return adapter.list_voices()


@router.post("/v1/voices", status_code=200)
async def register_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form(default=""),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_adapter()
    return adapter.register_voice(
        name=name,
        audio_data=audio_data,
        language=language,
        content_type=file.content_type or "audio/wav",
    )


@router.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    adapter = _resolve_adapter()
    deleted = adapter.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    return Response(status_code=204)
