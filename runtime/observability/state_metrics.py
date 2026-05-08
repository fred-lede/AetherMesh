from __future__ import annotations

import time
from typing import Any

from runtime.observability.metrics import metrics_collector


class StateMetricsCollector:
    def __init__(self) -> None:
        self._transition_counts: dict[str, int] = {}
        self._current_states: dict[str, str] = {}

    def record_transition(
        self,
        execution_id: str,
        source_type: str,
        from_state: str,
        to_state: str,
    ) -> None:
        key = f"{source_type}:{from_state}->{to_state}"
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
        self._current_states[execution_id] = to_state
        metrics_collector.increment(f"state.transition.{key}")
        metrics_collector.set_gauge(f"state.{source_type}.{to_state}", 1)

    def get_transition_count(
        self,
        source_type: str = "",
        from_state: str = "",
        to_state: str = "",
    ) -> int:
        total = 0
        for key, count in self._transition_counts.items():
            if source_type and not key.startswith(source_type):
                continue
            if from_state and from_state not in key:
                continue
            if to_state and f"->{to_state}" not in key:
                continue
            total += count
        return total

    def snapshot(self) -> dict[str, Any]:
        state_distribution: dict[str, int] = {}
        for state in self._current_states.values():
            state_distribution[state] = state_distribution.get(state, 0) + 1
        return {
            "total_transitions": sum(self._transition_counts.values()),
            "active_executions": len(self._current_states),
            "state_distribution": state_distribution,
            "transition_summary": dict(
                sorted(self._transition_counts.items(), key=lambda x: -x[1])[:20]
            ),
        }


state_metrics = StateMetricsCollector()
