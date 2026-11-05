from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings
from runtime.asr.asr_manager import asr_manager
from runtime.realtime.audio_pipeline import RealtimeAudioPipeline
from runtime.realtime.openai_events import (
    anchor_response_sequence,
    audio_transcript_delta_event,
    audio_transcript_done_event,
    input_item_added_event,
    input_item_created_event,
    is_anchor_request,
    new_event_id,
    response_sequence_for_text,
    transcription_completed_event,
)
from runtime.realtime.realtime_session import RealtimeSession
from runtime.security.database import SessionLocal
from runtime.security.auth.api_key import validate_api_key

logger = logging.getLogger("router.realtime_router")

router = APIRouter()


def preload_asr() -> None:
    asr_manager.preload()


async def _verify_ws_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    env_keys = __import__("os").getenv("AIIH_API_KEY", "").strip()
    if env_keys and api_key in [k.strip() for k in env_keys.split(",") if k.strip()]:
        return True
    try:
        db = SessionLocal()
        try:
            return validate_api_key(db, api_key) is not None
        finally:
            db.close()
    except Exception:
        return False


def _chunk_text(text: str, size: int = 40) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i : i + size]))
    return chunks or [""]


AUDIO_APPEND_TYPES = {"session.input_audio_buffer.append", "input_audio_buffer.append"}


def openai_realtime_extras(event: dict, *, audio_transcripts: bool = False) -> list[dict]:
    event_type = event.get("type")
    if event_type == "session.input_transcript.delta":
        text = str(event.get("text") or event.get("delta") or "").strip()
        if not text:
            return []
        item_id = f"item_{event.get('seg', 0)}"
        return [
            input_item_created_event(item_id),
            input_item_added_event(item_id),
            transcription_completed_event(item_id, text),
        ]
    if event_type == "session.output_transcript.delta":
        text = str(event.get("text") or event.get("delta") or "").strip()
        if not text:
            return []
        seg = event.get("seg", 0)
        return response_sequence_for_text(
            text,
            response_id=f"resp_{seg}",
            item_id=f"msg_{seg}",
            conversation_id="conv_local",
            previous_item_id=f"item_{seg}",
            audio_transcripts=audio_transcripts,
        )
    return []


async def _run_completion(session: RealtimeSession, service) -> dict:
    messages = session.build_messages()
    if not messages:
        raise ValueError("conversation is empty; send a conversation.item.create before response.create")
    payload = {
        "model": session.session.get("model") or "",
        "messages": messages,
        "stream": False,
        "max_output_tokens": session.session.get("max_output_tokens", 1024),
        "temperature": session.session.get("temperature", 0.8),
    }
    if not payload["model"]:
        raise ValueError("no model configured; send a session.update with a 'model'")
    result = await asyncio.to_thread(service.handle_chat, payload)
    choices = result.get("choices") or []
    if not choices:
        raise ValueError("completion returned no choices")
    content = (choices[0].get("message") or {}).get("content")
    return str(content or "")


async def _stream_response(websocket: WebSocket, session: RealtimeSession, service, send_json=None, *, metadata=None) -> None:
    send = send_json or websocket.send_json
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    echoed_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    response_body = {"id": response_id, "conversation_id": "conv_local", "metadata": echoed_metadata}
    previous_item_id = session.items[-1].get("id") if session.items else None
    try:
        await send({"type": "response.created", "event_id": new_event_id(), "response": {**response_body, "status": "in_progress"}})
        await send({
            "type": "response.output_item.added",
            "event_id": new_event_id(),
            "response_id": response_id,
            "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
        })
        await send({
            "type": "conversation.item.created",
            "event_id": new_event_id(),
            "previous_item_id": previous_item_id,
            "item": {"id": item_id, "object": "realtime.item", "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        })
        await send({
            "type": "response.content_part.added",
            "event_id": new_event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "part": {"type": "text", "text": ""},
        })
        text = await _run_completion(session, service)
        audio_transcripts = session.wants_audio_transcripts()
        for chunk in _chunk_text(text):
            await send({
                "type": "response.text.delta",
                "event_id": new_event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "delta": chunk,
            })
            if audio_transcripts:
                await send(audio_transcript_delta_event(chunk, response_id=response_id, item_id=item_id))
            else:
                await send({
                    "type": "response.output_text.delta",
                    "event_id": new_event_id(),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": chunk,
                })
            await asyncio.sleep(0.01)
        await send({"type": "response.text.done", "event_id": new_event_id(), "response_id": response_id, "item_id": item_id, "text": text})
        if audio_transcripts:
            await send(audio_transcript_done_event(text, response_id=response_id, item_id=item_id))
        else:
            await send({
                "type": "response.output_text.done",
                "event_id": new_event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            })
        await send({
            "type": "response.output_item.done",
            "event_id": new_event_id(),
            "response_id": response_id,
            "item": {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]},
        })
        await send({"type": "response.done", "event_id": new_event_id(), "response": {**response_body, "status": "completed"}})
        session.items.append({
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        })
    except asyncio.CancelledError:
        await send({"type": "response.done", "event_id": new_event_id(), "response": {**response_body, "status": "cancelled"}})
    except Exception as exc:
        await send({"type": "response.done", "event_id": new_event_id(), "response": {**response_body, "status": "failed", "error": {"message": str(exc)}}})


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket, service=None) -> None:
    from runtime.orchestration.openai_handler import RouterService

    if service is None:
        service = RouterService()

    protocols_header = websocket.headers.get("sec-websocket-protocol", "")
    proposed_subprotocol = ""
    for part in protocols_header.split(","):
        if part.strip():
            proposed_subprotocol = part.strip()
            break
    await websocket.accept(subprotocol=proposed_subprotocol or None)

    api_key = websocket.query_params.get("api_key", "")
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = api_key or auth_header[7:]
    for part in protocols_header.split(","):
        candidate = part.strip()
        if candidate.startswith("openai-insecure-api-key."):
            api_key = api_key or candidate[len("openai-insecure-api-key."):]
            break
    if not await _verify_ws_api_key(api_key):
        await websocket.send_json({"type": "error", "error": {"type": "authentication_error", "message": "Authentication failed"}})
        await websocket.close(code=4001)
        return

    session = RealtimeSession(model=websocket.query_params.get("model", ""))
    await websocket.send_json({"type": "session.created", "event_id": new_event_id(), "session": dict(session.session)})
    _t0 = time.monotonic()
    audio_pipeline: RealtimeAudioPipeline | None = None
    audio_queue: asyncio.Queue[tuple[int, bytes] | None] | None = None
    audio_task: asyncio.Task | None = None
    audio_started = False
    audio_cancel_pending = False
    audio_generation = 0
    response_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def send_json(event: dict) -> None:
        async with send_lock:
            await websocket.send_json(event)

    async def emit_with_extras(outbound: dict) -> None:
        await send_json(outbound)
        for extra in openai_realtime_extras(outbound, audio_transcripts=session.wants_audio_transcripts()):
            await send_json(extra)

    AUDIO_IDLE_POLL_SECONDS = 1.0

    async def process_audio(queue: asyncio.Queue[tuple[int, bytes] | None]) -> None:
        nonlocal audio_pipeline
        manager_version = asr_manager.version()
        while True:
            generation = audio_generation

            async def emit(outbound: dict) -> None:
                if generation != audio_generation:
                    return
                await emit_with_extras(outbound)

            try:
                item = await asyncio.wait_for(queue.get(), timeout=AUDIO_IDLE_POLL_SECONDS)
            except asyncio.TimeoutError:
                if audio_pipeline is not None and asr_manager.version() == manager_version:
                    try:
                        await audio_pipeline.flush_idle(emit=emit)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if generation == audio_generation:
                            await send_json({"type": "error", "error": {"type": "server_error", "message": str(exc)}})
                            audio_pipeline = None
                continue
            if item is None:
                queue.task_done()
                return
            generation, pcm = item
            try:
                if audio_pipeline is None or asr_manager.version() != manager_version:
                    audio_pipeline = None
                    adapter = await asyncio.to_thread(asr_manager.get_adapter)
                    if adapter is None:
                        raise RuntimeError("ASR adapter unavailable")
                    audio_pipeline = RealtimeAudioPipeline(
                        asr_model=adapter._model,
                        service=service,
                        model=session.session.get("model") or "",
                        target_language=session.session["target_language"],
                        instructions=str(session.session.get("instructions") or ""),
                    )
                    manager_version = asr_manager.version()
                await audio_pipeline.add_audio(pcm, emit=emit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if generation == audio_generation:
                    await send_json({"type": "error", "error": {"type": "server_error", "message": str(exc)}})
                    audio_pipeline = None
            finally:
                queue.task_done()

    async def stop_audio_processing(*, cancel: bool = False, wait: bool = True) -> None:
        nonlocal audio_queue, audio_task, audio_pipeline, audio_cancel_pending, audio_generation
        if audio_task is not None:
            if cancel:
                audio_generation += 1
                while True:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                audio_queue.put_nowait(None)
                audio_cancel_pending = True
                if not wait:
                    return
            elif audio_cancel_pending:
                if not wait:
                    return
            else:
                await audio_queue.put(None)
            if wait:
                await asyncio.gather(audio_task, return_exceptions=True)
                audio_queue = None
                audio_task = None
                if audio_cancel_pending:
                    audio_pipeline = None
                    audio_cancel_pending = False
            audio_queue = None
            audio_task = None

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("text")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "error": {"type": "invalid_request_error", "message": "message is not valid JSON"}})
                continue
            if not isinstance(data, dict):
                await send_json({"type": "error", "error": {"type": "invalid_request_error", "message": "message must be a JSON object"}})
                continue
            event_type = data.get("type")
            if not isinstance(event_type, str) or not event_type:
                await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "event type must be a string"}})
                continue
            if event_type == "response.create":
                response_body = data.get("response")
                if is_anchor_request(response_body):
                    anchor_events = anchor_response_sequence(f"resp_{uuid.uuid4().hex[:24]}", response_body.get("metadata") if isinstance(response_body, dict) else None)
                    for anchor_event in anchor_events:
                        await send_json(anchor_event)
                    continue
                if response_task is not None and not response_task.done():
                    await send_json({
                        "type": "error",
                        "event_id": data.get("event_id"),
                        "error": {
                            "type": "invalid_request_error",
                            "code": "conversation_already_has_active_response",
                            "message": "a response is already in progress",
                        },
                    })
                    continue
                metadata = response_body.get("metadata") if isinstance(response_body, dict) else None
                response_task = asyncio.create_task(_stream_response(websocket, session, service, send_json, metadata=metadata))
                await asyncio.sleep(0)
                continue
            if event_type == "session.update" and audio_started:
                await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "session settings cannot change after audio input starts"}})
                continue
            if event_type == "response.cancel":
                if response_task is not None and not response_task.done():
                    response_task.cancel()
                    await asyncio.gather(response_task, return_exceptions=True)
                response_task = None
                await stop_audio_processing(cancel=True, wait=False)
                await send_json({"type": "response.cancelled"})
                continue
            if event_type == "session.close":
                session.handle_event(data)
                cancelled_audio = audio_cancel_pending
                await stop_audio_processing()
                if response_task is not None:
                    await response_task
                if audio_pipeline is not None and not cancelled_audio:
                    try:
                        await audio_pipeline.close(emit=emit_with_extras)
                    except Exception as exc:
                        await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "server_error", "message": str(exc)}})
                await send_json({"type": "session.closed"})
                lifetime_s = time.monotonic() - _t0
                logger.info("Realtime websocket session.close (lifetime=%.1fs)", lifetime_s)
                await websocket.close()
                return

            outbound_events = session.handle_event(data)
            for outbound in outbound_events:
                await send_json(outbound)
            if event_type in AUDIO_APPEND_TYPES:
                pcm = session.take_audio()
                if any(event.get("type") == "error" for event in outbound_events):
                    continue
                if not (session.session.get("target_language") or str(session.session.get("instructions") or "").strip()):
                    await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "session.update must set session.audio.output.language or instructions before audio input"}})
                    continue
                if not settings.asr_enabled:
                    await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "server_error", "message": "ASR is not enabled"}})
                    continue
                if audio_cancel_pending:
                    if audio_task is not None and not audio_task.done():
                        await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "server_error", "message": "cancelled audio processing is still stopping"}})
                        continue
                    await stop_audio_processing()
                if audio_queue is None:
                    audio_queue = asyncio.Queue(maxsize=8)
                    audio_task = asyncio.create_task(process_audio(audio_queue))
                await audio_queue.put((audio_generation, pcm))
                audio_started = True
                continue
            if event_type not in {"session.update", "conversation.item.create", "ping", "input_audio_buffer.commit", "input_audio_buffer.clear"}:
                await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": f"unsupported event type '{event_type}'"}})
    except WebSocketDisconnect as exc:
        lifetime_s = time.monotonic() - _t0
        logger.info(
            "Realtime websocket disconnected (lifetime=%.1fs, close_code=%s, reason=%s)",
            lifetime_s,
            getattr(exc, "code", None),
            getattr(exc, "reason", None),
        )
    finally:
        await stop_audio_processing(cancel=True)
        if response_task is not None and not response_task.done():
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
