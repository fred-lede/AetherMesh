from __future__ import annotations

import logging
import time
from typing import Any

from runtime.sessions.session_store import session_store

logger = logging.getLogger("memory.short_term")


class ShortTermMemory:
    def add_message(
        self, session_id: str, role: str, content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not session_store.get(session_id):
            session_store.create(session_id, client_type="memory")
        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            message["metadata"] = metadata
        session_store.append_message(session_id, message)

    def get_context(
        self, session_id: str, max_messages: int = 20,
    ) -> list[dict[str, Any]]:
        session = session_store.get(session_id)
        if not session:
            return []
        messages = getattr(session, "messages", [])
        return messages[-max_messages:]

    def message_count(self, session_id: str) -> int:
        session = session_store.get(session_id)
        if not session:
            return 0
        messages = getattr(session, "messages", [])
        return len(messages)

    def set_metadata(self, session_id: str, key: str, value: Any) -> None:
        session_store.set_memory(session_id, key, value)

    def get_metadata(self, session_id: str, key: str, default: Any = None) -> Any:
        session = session_store.get(session_id)
        if not session:
            return default
        return getattr(session, "metadata", {}).get(key, default)

    def clear(self, session_id: str) -> None:
        session_store.delete(session_id)
