from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from config.settings import settings
from providers.asr_base import ASRProviderError
from providers.streaming_asr import StreamingASR
from providers.tts_base import TTSProviderError
from runtime.orchestration.provider_router import adapter as get_adapter
from runtime.security.auth.api_key import validate_api_key
from runtime.security.database import SessionLocal

logger = logging.getLogger("router.audio")
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
    adapter = get_adapter("xtts")
    if adapter is None:
        raise HTTPException(status_code=503, detail="TTS adapter unavailable (model load failed or cooling down)")
    return adapter


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

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
    try:
        return adapter.register_voice(
            name=name,
            audio_data=audio_data,
            language=language,
            content_type=file.content_type or "audio/wav",
        )
    except TTSProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e


@router.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    adapter = _resolve_adapter()
    deleted = adapter.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    return Response(status_code=204)


@router.patch("/v1/voices/{voice_id}")
async def update_voice(voice_id: str, body: dict[str, Any]) -> dict[str, Any]:
    adapter = _resolve_adapter()
    try:
        return adapter.update_voice(voice_id, name=body.get("name"), language=body.get("language"))
    except TTSProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e


@router.get("/v1/voices/{voice_id}/preview")
async def preview_voice(voice_id: str) -> FileResponse:
    adapter = _resolve_adapter()
    try:
        ref_path = adapter.voice_ref_path(voice_id)
    except TTSProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="Reference audio not found")
    return FileResponse(str(ref_path), media_type="audio/wav")


# ── ASR ──────────────────────────────────────────────────────


def _resolve_asr_adapter():
    if not settings.asr_enabled:
        raise HTTPException(status_code=503, detail="ASR is not enabled")
    adapter = get_adapter("asr")
    if adapter is None:
        raise HTTPException(status_code=503, detail="ASR adapter unavailable (model load failed or cooling down)")
    return adapter


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_asr_adapter()
    try:
        return adapter.transcribe(
            audio=audio_data,
            task="transcribe",
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )
    except ASRProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
) -> dict[str, Any]:
    audio_data = await file.read()
    adapter = _resolve_asr_adapter()
    try:
        return adapter.transcribe(
            audio=audio_data,
            task="translate",
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )
    except ASRProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


def _verify_ws_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    env_keys = __import__("os").getenv("AIIH_API_KEY", "").strip()
    if env_keys:
        if api_key in [k.strip() for k in env_keys.split(",") if k.strip()]:
            return True
    try:
        db = SessionLocal()
        try:
            return validate_api_key(db, api_key) is not None
        finally:
            db.close()
    except Exception:
        return False


@router.websocket("/v1/audio/transcriptions/stream")
async def websocket_asr_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    api_key = websocket.query_params.get("api_key", "")
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = api_key or auth_header[7:]
    if not api_key or not _verify_ws_api_key(api_key):
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close(code=4001)
        return

    language = websocket.query_params.get("language", "")
    interim = websocket.query_params.get("interim", "false").lower() == "true"

    if not settings.asr_enabled:
        await websocket.send_json({"type": "error", "message": "ASR is not enabled"})
        await websocket.close()
        return

    adapter = get_adapter("asr")
    if adapter is None:
        await websocket.send_json({"type": "error", "message": "ASR adapter unavailable"})
        await websocket.close()
        return

    stream = StreamingASR(
        model=adapter._model,
        language=language,
        interim=interim,
    )

    try:
        while not stream._closed:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                results = await stream.transcribe_if_ready()
                for r in results:
                    await websocket.send_json(r)
                continue

            if msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "flush":
                    results = await stream.flush()
                    for r in results:
                        await websocket.send_json(r)
                    break
                elif data.get("type") == "config":
                    if "language" in data:
                        stream.set_language(data["language"])
                    elif "lang" in data:
                        stream.set_language(data["lang"])
                    if "interim" in data:
                        stream._interim = bool(data["interim"])
            elif msg.get("bytes"):
                await stream.add_audio(msg["bytes"])
                results = await stream.transcribe_if_ready()
                for r in results:
                    await websocket.send_json(r)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket ASR error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        stream._closed = True
        try:
            await websocket.close()
        except Exception:
            pass
