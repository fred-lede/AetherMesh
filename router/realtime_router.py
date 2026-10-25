from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings
from runtime.orchestration.provider_router import adapter as get_adapter
from runtime.realtime.audio_pipeline import RealtimeAudioPipeline
from runtime.realtime.realtime_session import RealtimeSession
from runtime.security.database import SessionLocal
from runtime.security.auth.api_key import validate_api_key

logger = logging.getLogger("router.realtime_router")

router = APIRouter()


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


async def _stream_response(websocket: WebSocket, session: RealtimeSession, service, send_json=None) -> None:
    send = send_json or websocket.send_json
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    try:
        await send({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})
        await send({
            "type": "response.output_item.added",
            "response_id": response_id,
            "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
        })
        await send({
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "part": {"type": "text", "text": ""},
        })
        text = await _run_completion(session, service)
        for chunk in _chunk_text(text):
            await send({
                "type": "response.text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "delta": chunk,
            })
            await asyncio.sleep(0.01)
        await send({"type": "response.text.done", "response_id": response_id, "item_id": item_id, "text": text})
        await send({
            "type": "response.output_item.done",
            "response_id": response_id,
            "item": {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]},
        })
        await send({"type": "response.done", "response": {"id": response_id, "status": "completed"}})
        session.items.append({
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        })
    except asyncio.CancelledError:
        await send({"type": "response.done", "response": {"id": response_id, "status": "cancelled"}})
    except Exception as exc:
        await send({"type": "response.done", "response": {"id": response_id, "status": "failed", "error": {"message": str(exc)}}})


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket, service=None) -> None:
    from runtime.orchestration.openai_handler import RouterService

    if service is None:
        service = RouterService()
    await websocket.accept()

    api_key = websocket.query_params.get("api_key", "")
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = api_key or auth_header[7:]
    if not await _verify_ws_api_key(api_key):
        await websocket.send_json({"type": "error", "error": {"type": "authentication_error", "message": "Authentication failed"}})
        await websocket.close(code=4001)
        return

    session = RealtimeSession(model=websocket.query_params.get("model", ""))
    await websocket.send_json({"type": "session.created", "session": dict(session.session)})
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

    async def process_audio(queue: asyncio.Queue[tuple[int, bytes] | None]) -> None:
        nonlocal audio_pipeline
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            generation, pcm = item
            try:
                if audio_pipeline is None:
                    asr_adapter = await asyncio.to_thread(get_adapter, "asr")
                    if asr_adapter is None:
                        raise RuntimeError("ASR adapter unavailable")
                    audio_pipeline = RealtimeAudioPipeline(
                        asr_model=asr_adapter._model,
                        service=service,
                        model=session.session.get("model") or "",
                        target_language=session.session["target_language"],
                    )
                async def emit(outbound: dict) -> None:
                    if generation == audio_generation:
                        await send_json(outbound)

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
                if response_task is not None and not response_task.done():
                    await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "a response is already in progress"}})
                    continue
                response_task = asyncio.create_task(_stream_response(websocket, session, service, send_json))
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
                        await audio_pipeline.close(emit=send_json)
                    except Exception as exc:
                        await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "server_error", "message": str(exc)}})
                await send_json({"type": "session.closed"})
                await websocket.close()
                return

            outbound_events = session.handle_event(data)
            for outbound in outbound_events:
                await send_json(outbound)
            if event_type == "session.input_audio_buffer.append":
                pcm = session.take_audio()
                if any(event.get("type") == "error" for event in outbound_events):
                    continue
                if not session.session.get("target_language"):
                    await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "session.update must set session.audio.output.language before audio input"}})
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
            if event_type not in {"session.update", "conversation.item.create", "ping", "input_audio_buffer.append", "input_audio_buffer.commit"}:
                await send_json({"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": f"unsupported event type '{event_type}'"}})
    except WebSocketDisconnect:
        logger.info("Realtime websocket disconnected")
    finally:
        await stop_audio_processing(cancel=True)
        if response_task is not None and not response_task.done():
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
