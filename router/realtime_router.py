from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings
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


async def _stream_response(websocket: WebSocket, session: RealtimeSession, service) -> None:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    await websocket.send_json({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})
    await websocket.send_json({
        "type": "response.output_item.added",
        "response_id": response_id,
        "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
    })
    await websocket.send_json({
        "type": "response.content_part.added",
        "response_id": response_id,
        "item_id": item_id,
        "part": {"type": "text", "text": ""},
    })
    try:
        text = await _run_completion(session, service)
    except Exception as exc:
        await websocket.send_json({"type": "response.done", "response": {"id": response_id, "status": "failed", "error": {"message": str(exc)}}})
        return
    for chunk in _chunk_text(text):
        await websocket.send_json({
            "type": "response.text.delta",
            "response_id": response_id,
            "item_id": item_id,
            "delta": chunk,
        })
        await asyncio.sleep(0.01)
    await websocket.send_json({"type": "response.text.done", "response_id": response_id, "item_id": item_id, "text": text})
    await websocket.send_json({
        "type": "response.output_item.done",
        "response_id": response_id,
        "item": {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]},
    })
    await websocket.send_json({"type": "response.done", "response": {"id": response_id, "status": "completed"}})
    session.items.append({
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    })


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
                await websocket.send_json({"type": "error", "error": {"type": "invalid_request_error", "message": "message is not valid JSON"}})
                continue
            event_type = data.get("type")
            if event_type == "response.create":
                await _stream_response(websocket, session, service)
                continue
            if event_type == "response.cancel":
                await websocket.send_json({"type": "response.cancelled"})
                continue
            for outbound in session.handle_event(data):
                await websocket.send_json(outbound)
    except WebSocketDisconnect:
        logger.info("Realtime websocket disconnected")
