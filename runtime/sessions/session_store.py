from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from config.settings import settings

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        data = dict(data)
        data["id"] = str(data.get("id", ""))
        data["messages"] = data.get("messages") or []
        data["agent_memory"] = data.get("agent_memory") or {}
        data["tools"] = data.get("tools") or []
        data["metadata"] = data.get("metadata") or {}
        return cls(**data)


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or settings.config_path("sessions.json")
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable session store %s", self._path)
            return
        if isinstance(data, dict):
            for sid, raw in (data.get("sessions") or {}).items():
                if isinstance(raw, dict):
                    self._sessions[sid] = Session.from_dict(raw)

    def _save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"sessions": {sid: s.to_dict() for sid, s in self._sessions.items()}}, ensure_ascii=False),
                encoding="utf-8",
            )

    def create(self, session_id: str, client_type: str = "unknown", ttl_s: int = 3600) -> Session:
        with self._lock:
            session = Session(
                id=session_id,
                client_type=client_type,
                created_at=time.time(),
                expires_at=time.time() + ttl_s if ttl_s > 0 else 0,
            )
            self._sessions[session_id] = session
            self._save()
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.expired:
                self._sessions.pop(session_id, None)
                self._save()
                return None
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
            if removed:
                self._save()
            return removed

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.expired:
                session.messages.append(message)
                self._save()

    def set_memory(self, session_id: str, key: str, value: Any) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.agent_memory[key] = value
                self._save()

    def get_memory(self, session_id: str, key: str, default: Any = None) -> Any:
        session = self.get(session_id)
        return session.agent_memory.get(key, default) if session else default

    def update_metadata(self, session_id: str, updates: dict[str, Any]) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.expired:
                return None
            session.metadata.update(updates)
            self._save()
            return session

    def get_context(self, session_id: str, max_messages: int = 20) -> list[dict[str, Any]]:
        session = self.get(session_id)
        if session is None:
            return []
        return session.messages[-max_messages:]

    def list_active(self) -> list[Session]:
        with self._lock:
            now = time.time()
            return [s for s in self._sessions.values() if s.expires_at <= 0 or s.expires_at > now]

    def evict_expired(self) -> int:
        with self._lock:
            now = time.time()
            expired = [sid for sid, s in self._sessions.items() if 0 < s.expires_at < now]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                self._save()
            return len(expired)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def summarize_and_trim(
        self,
        session_id: str,
        summarizer: Callable[[str], str],
        threshold: int = 40,
        keep_last: int = 10,
    ) -> Session | None:
        session = self.get(session_id)
        if session is None:
            return None
        if session.message_count < threshold:
            return session
        transcript = "\n".join(
            f"{m.get('role', 'unknown')}: {_message_text(m)}"
            for m in session.messages
            if m.get("content")
        )
        try:
            summary = summarizer(transcript)
        except Exception as exc:
            logger.warning("Session summarization failed for %s: %s", session_id, exc)
            return session
        summary = str(summary or "").strip()
        with self._lock:
            if not summary:
                return session
            prior = session.metadata.get("summary")
            combined = f"{prior}\n{summary}" if prior else summary
            session.metadata["summary"] = combined
            kept = session.messages[-keep_last:]
            if kept and kept[0].get("role") == "system":
                kept = kept[1:]
            session.messages = [
                {"role": "system", "content": f"[Prior conversation summary]\n{combined}"},
                *kept,
            ]
            self._save()
        return session


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content[:1000]
    if isinstance(content, list):
        parts = [
            str(p.get("text", ""))[:1000]
            for p in content
            if isinstance(p, dict) and p.get("type") in {"text", "input_text"}
        ]
        return " ".join(parts)[:1000]
    return str(content)[:1000]


session_store = SessionStore()
