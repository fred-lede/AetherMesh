from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("security.audit")


class AuditLog:
    def __init__(self, path: str | None = None) -> None:
        if path:
            self._path = Path(path)
        else:
            self._path = Path(os.getcwd()) / "config" / "security_audit.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, details: dict[str, Any], actor: str = "system") -> None:
        event = {
            "timestamp": time.time(),
            "actor": actor,
            "action": action,
            "details": details,
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)

    def record_tool_execution(
        self, tool_name: str, tool_id: str, actor: str, duration_ms: float, is_error: bool
    ) -> None:
        self.record(
            "tool_execution",
            {"tool_name": tool_name, "tool_id": tool_id, "duration_ms": round(duration_ms, 1), "is_error": is_error},
            actor=actor,
        )

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        events = self.query(limit=0)
        return events[-max(1, limit):]

    def query(
        self,
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
        actor: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        details_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if action and item.get("action") != action:
                continue
            if actor and item.get("actor") != actor:
                continue
            ts = item.get("timestamp")
            if isinstance(ts, (int, float)):
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue
            if details_filter:
                details = item.get("details") or {}
                if not all(details.get(k) == v for k, v in details_filter.items()):
                    continue
            events.append(item)
        return events[offset : offset + limit] if limit > 0 else events[offset:]


audit_log = AuditLog()
