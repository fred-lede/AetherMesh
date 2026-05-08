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
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events


audit_log = AuditLog()
