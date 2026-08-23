from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("memory.episodic")

_MAX_RECORDS = 5000


@dataclass
class EpisodicRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str = ""
    model: str = ""
    provider: str = ""
    task_summary: str = ""
    tool_calls: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    token_count: dict[str, int] | None = None


class EpisodicMemory:
    def __init__(self, max_records: int = _MAX_RECORDS) -> None:
        self._records: deque[EpisodicRecord] = deque(maxlen=max_records)

    def record(self, **kwargs: Any) -> EpisodicRecord:
        record = EpisodicRecord(**kwargs)
        self._records.append(record)
        logger.debug("Recorded episodic: %s/%s (%dms)", record.provider, record.model, record.duration_ms)
        return record

    def by_session(self, session_id: str, limit: int = 20) -> list[EpisodicRecord]:
        matched = [r for r in self._records if r.session_id == session_id]
        return matched[-limit:]

    def by_provider(self, provider: str, limit: int = 20) -> list[EpisodicRecord]:
        matched = [r for r in self._records if r.provider == provider]
        return matched[-limit:]

    def by_model(self, model: str, limit: int = 20) -> list[EpisodicRecord]:
        matched = [r for r in self._records if r.model == model]
        return matched[-limit:]

    def recent(self, limit: int = 10) -> list[EpisodicRecord]:
        if limit >= len(self._records):
            return list(self._records)
        start = len(self._records) - limit
        return [self._records[i] for i in range(start, len(self._records))]

    def failures(self, limit: int = 10) -> list[EpisodicRecord]:
        return [r for r in self._records if not r.success][-limit:]

    def success_rate(self, provider: str | None = None) -> float:
        records = self._records
        if provider:
            records = [r for r in records if r.provider == provider]
        if not records:
            return 1.0
        successes = sum(1 for r in records if r.success)
        return successes / len(records)

    def avg_latency_ms(self, provider: str) -> float:
        records = [r for r in self._records if r.provider == provider and r.success]
        if not records:
            return 0.0
        return sum(r.duration_ms for r in records) / len(records)

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)
