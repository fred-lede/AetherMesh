from __future__ import annotations

import time
from typing import Any

from runtime.observability.metrics import metrics_collector


class ReplayMetricsCollector:
    def __init__(self) -> None:
        self._replay_count: int = 0
        self._recorded_count: int = 0

    def record_replay(self, execution_id: str, event_count: int) -> None:
        self._replay_count += 1
        metrics_collector.increment("replay.executions")
        metrics_collector.record("replay.events_per_execution", event_count)

    def record_recording(self, execution_id: str, event_count: int) -> None:
        self._recorded_count += 1
        metrics_collector.increment("replay.recordings")
        metrics_collector.record("replay.recorded_events", event_count)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_replays": self._replay_count,
            "total_recordings": self._recorded_count,
            "replay_count": metrics_collector.get_counter("replay.executions"),
        }


replay_metrics = ReplayMetricsCollector()
