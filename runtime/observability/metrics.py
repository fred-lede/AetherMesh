from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(self, metric: str, value: int = 1) -> None:
        self._counters[metric] += value

    def record(self, metric: str, value: float) -> None:
        self._histograms[metric].append(value)

    def set_gauge(self, metric: str, value: float) -> None:
        self._gauges[metric] = value

    def get_counter(self, metric: str) -> int:
        return self._counters.get(metric, 0)

    def get_histogram(self, metric: str) -> dict[str, float]:
        vals = self._histograms.get(metric, [])
        if not vals:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(vals),
            "sum": sum(vals),
            "avg": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    def get_gauge(self, metric: str) -> float:
        return self._gauges.get(metric, 0.0)

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in self._counters.items():
            result[f"counter:{k}"] = v
        for k in self._histograms:
            result[f"histogram:{k}"] = self.get_histogram(k)
        for k, v in self._gauges.items():
            result[f"gauge:{k}"] = v
        return result

    def clear(self) -> None:
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()


metrics_collector = MetricsCollector()
