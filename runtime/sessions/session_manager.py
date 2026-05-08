from __future__ import annotations

import logging
import time
from typing import Any

from runtime.sessions.session_store import Session, session_store

logger = logging.getLogger("sessions.manager")


class SessionManager:
    def __init__(self) -> None:
        self._client_session_map: dict[str, str] = {}

    def get_or_create(self, client_id: str, client_type: str = "unknown", ttl_s: int = 3600) -> Session:
        existing_id = self._client_session_map.get(client_id)
        if existing_id:
            session = session_store.get(existing_id)
            if session:
                return session
        session = session_store.create(f"ses_{client_id}_{int(time.time())}", client_type=client_type, ttl_s=ttl_s)
        self._client_session_map[client_id] = session.id
        return session

    def get_session(self, session_id: str) -> Session | None:
        return session_store.get(session_id)

    def add_message(self, session_id: str, role: str, content: Any) -> None:
        session_store.append_message(session_id, {"role": role, "content": content, "timestamp": time.time()})

    def get_context(self, session_id: str, max_messages: int = 20) -> list[dict[str, Any]]:
        session = session_store.get(session_id)
        if not session:
            return []
        return session.messages[-max_messages:]

    def close_session(self, client_id: str) -> None:
        session_id = self._client_session_map.pop(client_id, None)
        if session_id:
            session_store.delete(session_id)

    def active_count(self) -> int:
        return session_store.count()


session_manager = SessionManager()
