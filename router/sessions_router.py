from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from config.settings import settings
from runtime.sessions.session_store import session_store

logger = logging.getLogger("router.sessions_router")


def _make_summarizer(service):
    def summarize(transcript: str) -> str:
        payload = {
            "model": settings.session_summary_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation into a concise summary in the "
                        "same language as the conversation. Preserve key facts, decisions, and "
                        "user preferences. Output only the summary."
                    ),
                },
                {"role": "user", "content": transcript[-12000:]},
            ],
            "stream": False,
        }
        result = service.handle_chat(payload)
        choices = result.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else ""
        return str(content or "")

    return summarize


def create_sessions_router(service: Any):
    router = APIRouter(prefix="/v1/sessions")
    summarize = _make_summarizer(service)

    @router.post("")
    def create_session(payload: dict[str, Any] = Body(default={})):
        session = session_store.create(
            session_id=f"ses_{uuid.uuid4().hex[:24]}",
            client_type=str(payload.get("client_type", "unknown")),
            ttl_s=int(payload.get("ttl_s", 3600)),
        )
        if isinstance(payload.get("metadata"), dict):
            session_store.update_metadata(session.id, payload["metadata"])
        return session_store.get(session.id).to_dict()

    @router.get("")
    def list_sessions(limit: int = 100):
        sessions = session_store.list_active()[-max(1, min(limit, 500)):]
        return {
            "object": "list",
            "data": [s.to_dict() for s in sessions],
            "count": len(sessions),
        }

    @router.get("/{session_id}")
    def get_session(session_id: str):
        session = session_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return session.to_dict()

    @router.patch("/{session_id}")
    def update_session(session_id: str, payload: dict[str, Any] = Body(...)):
        updates = payload.get("metadata")
        if not isinstance(updates, dict):
            raise HTTPException(status_code=400, detail="Expected 'metadata' object")
        session = session_store.update_metadata(session_id, updates)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return session.to_dict()

    @router.delete("/{session_id}")
    def delete_session(session_id: str):
        if not session_store.delete(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {"deleted": True}

    @router.get("/{session_id}/messages")
    def get_messages(session_id: str, max_messages: int = 50):
        session = session_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {
            "object": "list",
            "data": session_store.get_context(session_id, max_messages=max_messages),
        }

    @router.post("/{session_id}/messages")
    def append_message(session_id: str, payload: dict[str, Any] = Body(...)):
        role = payload.get("role")
        content = payload.get("content")
        if role not in {"user", "assistant", "system", "tool"}:
            raise HTTPException(status_code=400, detail="Invalid 'role'")
        if not content:
            raise HTTPException(status_code=400, detail="Missing 'content'")
        if session_store.get(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        session_store.append_message(session_id, {"role": role, "content": content, "timestamp": __import__("time").time()})
        session = session_store.get(session_id)
        if session.message_count >= settings.session_summarize_threshold:
            session_store.summarize_and_trim(
                session_id,
                summarize,
                threshold=settings.session_summarize_threshold,
            )
        return session_store.get(session_id).to_dict()

    @router.post("/{session_id}/summarize")
    def summarize_session(
        session_id: str,
        payload: dict[str, Any] = Body(default={}),
    ):
        if session_store.get(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        session = session_store.summarize_and_trim(
            session_id,
            summarize,
            threshold=int(payload.get("threshold", settings.session_summarize_threshold)),
            keep_last=int(payload.get("keep_last", 10)),
        )
        return session.to_dict()

    return router
