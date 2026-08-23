from __future__ import annotations

import time
from collections import deque
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.bus import runtime_event_bus
from runtime.events.event_types import EventType
from runtime.observability.metrics import metrics_collector

_MAX_DURATION_SAMPLES = 1000


class EventMetricsCollector:
    def __init__(self) -> None:
        self._event_counts: dict[str, int] = {}
        self._event_durations: dict[str, deque[float]] = {}
        self._start_time: float = time.time()

    def record_event(self, event: RuntimeEvent) -> None:
        type_name = event.type_name
        self._event_counts[type_name] = self._event_counts.get(type_name, 0) + 1
        metrics_collector.increment(f"events.{type_name}.count")
        if event.duration_ms > 0:
            bucket = self._event_durations.setdefault(
                type_name, deque(maxlen=_MAX_DURATION_SAMPLES)
            )
            bucket.append(event.duration_ms)
            metrics_collector.record(f"events.{type_name}.duration_ms", event.duration_ms)

    def get_count(self, event_type: EventType | str = "") -> int:
        if isinstance(event_type, EventType):
            key = event_type.value
        elif event_type:
            key = event_type
        else:
            return sum(self._event_counts.values())
        return self._event_counts.get(key, 0)

    def avg_duration(self, event_type: EventType | str) -> float:
        key = event_type.value if isinstance(event_type, EventType) else event_type
        durations = self._event_durations.get(key)
        return (sum(durations) / len(durations)) if durations else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_events": sum(self._event_counts.values()),
            "event_types": len(self._event_counts),
            "by_type": dict(self._event_counts),
            "uptime_s": time.time() - self._start_time,
        }

    def clear(self) -> None:
        self._event_counts.clear()
        self._event_durations.clear()


event_metrics = EventMetricsCollector()


async def _event_metrics_subscriber(event: RuntimeEvent) -> None:
    event_metrics.record_event(event)


runtime_event_bus.subscribe(_event_metrics_subscriber, name="event_metrics")
