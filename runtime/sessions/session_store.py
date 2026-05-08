from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("sessions.store")


@dataclass
class Session:
    id: str
    client_type: str = "unknown"
    messages: list[dict[str, Any]] = field(default_factory=list)
    agent_memory: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    @property
    def expired(self) -> bool:
        return 0 < self.expires_at < time.time()

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def message_count(self) -> int:
        return len(self.messages)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str, client_type: str = "unknown", ttl_s: int = 3600) -> Session:
        session = Session(
            id=session_id,
            client_type=client_type,
            created_at=time.time(),
            expires_at=time.time() + ttl_s if ttl_s > 0 else 0,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session and session.expired:
            self._sessions.pop(session_id, None)
            return None
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        session = self.get(session_id)
        if session:
            session.messages.append(message)

    def set_memory(self, session_id: str, key: str, value: Any) -> None:
        session = self.get(session_id)
        if session:
            session.agent_memory[key] = value

    def get_memory(self, session_id: str, key: str, default: Any = None) -> Any:
        session = self.get(session_id)
        return session.agent_memory.get(key, default) if session else default

    def list_active(self) -> list[Session]:
        now = time.time()
        return [s for s in self._sessions.values() if s.expires_at <= 0 or s.expires_at > now]

    def evict_expired(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if 0 < s.expires_at < now]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def count(self) -> int:
        return len(self._sessions)


session_store = SessionStore()
