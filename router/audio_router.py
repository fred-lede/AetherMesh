from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import httpx
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
    "aac": "audio/aac",
}


def _resolve_adapter():
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is not enabled")
    adapter = get_adapter("xtts")
    if adapter is None:
        raise HTTPException(status_code=503, detail="TTS adapter unavailable (model load failed or cooling down)")
    return adapter


TTS_MAX_CHARS: int = 150


def _truncate_for_tts(text: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else TTS_MAX_CHARS
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_boundary = max(truncated.rfind("。"), truncated.rfind("！"),
                        truncated.rfind("？"), truncated.rfind("."),
                        truncated.rfind("!"), truncated.rfind("?"))
    if last_boundary > limit // 2:
        return text[:last_boundary + 1]
    return truncated


def _ollama_base_url() -> str:
    try:
        base = getattr(settings, "ollama_fallback_base_url", None)()
        if base:
            return str(base).rstrip("/")
    except Exception:
        pass
    try:
        base = getattr(settings, "ollama_default_base_url", None)
        if callable(base):
            base = base()
        if base:
            return str(base).rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:11434"


OLLAMA_TTS_VOICE_FALLBACKS: dict[str, list[str]] = {
    "phonon": [
        "Alfie", "Azgorath", "Bryony", "Captain Silas Blackwater", "Delia",
        "Easton", "Edwin", "Elodie-Rose", "Everly", "Fairy Godmother",
        "Fennick the Young Elf", "Freya", "Grub the Ogre", "Hadley", "Harper",
        "Hugh", "Imogen", "Juniper", "Lottie", "Maeve", "Master Aldric",
        "Micah", "Miles", "Morwenna of the Deep", "Old Mortlake", "Pippa",
        "Preston", "Russell", "Sir Bramwell", "Spencer", "Teddy",
        "The Snow Queen", "The Thing Under the Stairs", "Toby", "Trevor",
        "Vyrmathrax the A",
    ],
}


_TTS_MODEL_KEYWORDS = ("phonon", "kokoro", "f5-tts", "f5tts", "voice", "speech")


def _ollama_tts_model_info(model: str) -> dict[str, Any] | None:
    """Match a model name against the registry; return {model, base_url} when it
    is an Ollama model with TTS voices.

    A model is treated as an Ollama TTS model when:
      - provider is 'ollama', AND
      - the name contains a well-known TTS model keyword, OR
      - the OLLAMA_TTS_VOICE_FALLBACKS table has an entry for it.

    This deliberately avoids using 'audio' capability, which is also set on
    Ollama models that merely *accept* audio input (e.g. gemma4)."""
    if not isinstance(model, str) or not model:
        return None
    try:
        from runtime.orchestration.model_registry_store import get_models

        mlow = model.lower()
        mstripped = mlow.removeprefix("ollama/")
        for entry in get_models():
            if str(entry.get("provider", "")).lower() != "ollama":
                continue
            name = str(entry.get("name", ""))
            if mlow != name.lower() and mstripped != name.lower():
                continue
            is_tts = any(kw in name.lower() for kw in _TTS_MODEL_KEYWORDS)
            is_tts = is_tts or (name in OLLAMA_TTS_VOICE_FALLBACKS)
            if not is_tts:
                continue
            for binding in entry.get("worker_bindings", []):
                if not isinstance(binding, dict):
                    continue
                base_url = settings.worker_base_url(binding)
                if base_url:
                    return {"model": name, "base_url": base_url.rstrip("/")}
    except Exception:
        return None
    return None


def _ollama_tts_voices(model: str, base_url: str) -> tuple[list[str], str]:
    """Best-effort voice list from the Ollama node, with hardcoded fallback."""
    try:
        r = httpx.get(f"{base_url}/v1/models/{model}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            candidates = []
            for key in ("voices", "voice", "voice_options"):
                if key in data:
                    candidates = data[key]
                    break
            if isinstance(candidates, list) and candidates:
                names = []
                for item in candidates:
                    if isinstance(item, dict):
                        names.append(str(item.get("name") or item.get("id") or ""))
                    else:
                        names.append(str(item))
                names = [n for n in names if n]
                if names:
                    return names, "ollama"
    except Exception:
        pass
    for key in (model, model.lower()):
        fallback = OLLAMA_TTS_VOICE_FALLBACKS.get(key)
        if fallback:
            return list(fallback), "fallback"
    return [], "unknown"


async def _create_speech_ollama(payload: dict[str, Any], info: dict[str, Any]) -> Response:
    model = info["model"]
    base_url = info["base_url"]

    text = payload.get("input", "") or ""
    max_chars = payload.get("max_chars")
    if max_chars is not None:
        try:
            text = _truncate_for_tts(text, int(max_chars))
        except (TypeError, ValueError):
            pass
    else:
        text = _truncate_for_tts(text, 500)
    if not text:
        raise HTTPException(status_code=422, detail="input is required")

    voice = str(payload.get("voice") or "")
    response_format = (payload.get("response_format") or "wav").lower()
    speed = payload.get("speed", 1.0)

    if response_format not in ("wav", "pcm"):
        raise HTTPException(
            status_code=400,
            detail=f"Ollama TTS only supports 'wav' or 'pcm' response_format, got '{response_format}'",
        )

    body: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": response_format,
    }
    if voice:
        body["voice"] = voice
    if speed is not None:
        try:
            body["speed"] = float(speed)
        except (TypeError, ValueError):
            pass

    url = f"{base_url}/v1/audio/speech"
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Ollama TTS request failed: {e}")

    if resp.status_code >= 400:
        detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=detail or "Ollama TTS error")

    content_type = resp.headers.get("content-type") or AUDIO_CONTENT_TYPES.get(response_format, "audio/wav")
    return Response(content=resp.content, media_type=content_type)


@router.get("/v1/tts/voices")
async def list_tts_voices(model: str = "") -> dict[str, Any]:
    """List voices for an Ollama TTS model (registry-resolved node)."""
    info = _ollama_tts_model_info(model) if model else None
    if info is None:
        raise HTTPException(status_code=404, detail=f"No Ollama TTS model found for '{model}'")
    voices, source = _ollama_tts_voices(info["model"], info["base_url"])
    return {
        "model": info["model"],
        "voices": voices,
        "source": source,
    }


@router.post("/v1/audio/speech")
async def create_speech(payload: dict[str, Any]) -> Response:
    model = payload.get("model") or ""
    info = _ollama_tts_model_info(model)
    if info is not None:
        return await _create_speech_ollama(payload, info)

    text = _truncate_for_tts(payload.get("input", ""))
    voice = payload.get("voice", "")
    response_format = payload.get("response_format", "wav")
    language = payload.get("language", "")
    speed = payload.get("speed", 1.0)
    max_chars = payload.get("max_chars")
    if max_chars is not None:
        try:
            text = _truncate_for_tts(payload.get("input", ""), int(max_chars))
        except (TypeError, ValueError):
            text = _truncate_for_tts(payload.get("input", ""))

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


def _resolve_voice_id(voice: str) -> str:
    if "/" in voice or "\\" in voice:
        return voice
    voices_dir = Path(settings.tts_voices_dir)
    if not voices_dir.is_dir():
        return voice
    for entry in sorted(voices_dir.iterdir()):
        meta_path = entry / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("name") == voice:
                    return str(meta.get("voice_id", entry.name))
            except Exception:
                pass
    return voice


@router.post("/v1/audio/chat")
async def audio_chat(
    file: UploadFile = File(...),
    model: str = Form("gemma4:e2b"),
    system_prompt: str = Form(default=""),
    voice: str = Form(default=""),
    language: str = Form(default=""),
    temperature: float = Form(default=0.0),
    max_tokens: int = Form(default=0),
    response_format: str = Form(default="json"),
    messages: str = Form(default=""),
) -> Any:
    audio_data = await file.read()

    # Step 1: ASR (speech → text)
    asr_adapter = _resolve_asr_adapter()
    try:
        transcript_result = asr_adapter.transcribe(
            audio=audio_data,
            task="transcribe",
            language=language,
            temperature=temperature,
        )
    except ASRProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    transcript = transcript_result.get("text", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="ASR returned empty transcript")

    # Step 2: LLM (text → response)
    from router.openai_router import service

    chat_messages: list[dict[str, Any]] = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    if messages:
        try:
            history = json.loads(messages)
            if isinstance(history, list):
                chat_messages.extend(history)
        except (json.JSONDecodeError, TypeError):
            pass
    chat_messages.append({"role": "user", "content": transcript})

    payload: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    try:
        result = service.handle_chat(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    response_text = ""
    if isinstance(result, dict):
        choices = result.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            response_text = message.get("content", "")

    if not response_text:
        raise HTTPException(status_code=500, detail="LLM returned empty response")

    # Step 3: TTS (text → speech, optional)
    audio_bytes: bytes | None = None
    if voice:
        resolved_voice = _resolve_voice_id(voice)
        tts_adapter = _resolve_adapter()
        tts_text = _truncate_for_tts(response_text)
        try:
            audio_bytes = tts_adapter.tts({
                "model": "xtts-v2",
                "input": tts_text,
                "voice": resolved_voice,
                "language": language or "",
                "speed": 1.0,
            })
        except TTSProviderError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e))
        if response_format in ("mp3", "opus", "flac"):
            audio_bytes = _convert_format(audio_bytes, response_format)

    # Step 4: return
    if response_format in ("wav", "mp3", "opus", "flac"):
        if audio_bytes:
            ct = AUDIO_CONTENT_TYPES.get(response_format, "audio/wav")
            return Response(content=audio_bytes, media_type=ct)
        return {"text": response_text, "transcript": transcript}

    result_data: dict[str, Any] = {
        "text": response_text,
        "transcript": transcript,
    }
    if audio_bytes:
        result_data["audio"] = base64.b64encode(audio_bytes).decode("ascii")

    return result_data


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
