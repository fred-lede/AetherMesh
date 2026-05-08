from __future__ import annotations

import logging
import time
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.events.bus import runtime_event_bus

logger = logging.getLogger("events.trace")


class EventTrace:
    def __init__(self, max_events: int = 5000) -> None:
        self._events: list[RuntimeEvent] = []
        self._max_events = max_events

    def record(self, event: RuntimeEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

    def get_trace(
        self,
        execution_id: str = "",
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[RuntimeEvent]:
        result = self._events
        if execution_id:
            result = [e for e in result if e.execution_id == execution_id]
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        return result[-limit:]

    def replay_events(self, events: list[RuntimeEvent]) -> None:
        for event in events:
            self.record(event)

    def clear(self) -> None:
        self._events.clear()

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "type": e.type_name,
                "timestamp": e.timestamp,
                "execution_id": e.execution_id,
                "source": e.source,
                "payload": e.payload,
                "error": e.error,
                "duration_ms": e.duration_ms,
            }
            for e in self._events
        ]


event_trace = EventTrace()


def _trace_subscriber(event: RuntimeEvent) -> None:
    event_trace.record(event)


runtime_event_bus.subscribe(_trace_subscriber, name="event_trace")
