from __future__ import annotations

import logging
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.replay.execution_snapshot import ExecutionSnapshot
from runtime.replay.recorder import RecordedExecution

logger = logging.getLogger("replay.trace_rebuilder")


class TraceRebuilder:
    def rebuild_from_events(
        self,
        events: list[RuntimeEvent],
        execution_id: str = "",
    ) -> list[dict[str, Any]]:
        spans: dict[str, dict[str, Any]] = {}
        for event in events:
            span_id = event.payload.get("span_id", "")
            if not span_id:
                continue
            if span_id not in spans:
                spans[span_id] = {
                    "span_id": span_id,
                    "name": event.payload.get("span_name", event.type_name),
                    "start_time": event.timestamp,
                    "end_time": event.timestamp,
                    "events": [],
                }
            spans[span_id]["events"].append(event.type_name)
            spans[span_id]["end_time"] = max(
                spans[span_id]["end_time"], event.timestamp,
            )

        return [
            {
                "span_id": sid,
                "name": s["name"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "duration_ms": (s["end_time"] - s["start_time"]) * 1000,
                "event_count": len(s["events"]),
                "events": s["events"],
            }
            for sid, s in sorted(
                spans.items(), key=lambda x: x[1]["start_time"],
            )
        ]

    def rebuild_timeline(
        self,
        recording: RecordedExecution,
    ) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for event in recording.events:
            timeline.append({
                "time_offset_ms": (event.timestamp - recording.start_time) * 1000,
                "type": event.type_name,
                "source": event.source,
                "duration_ms": event.duration_ms,
                "error": event.error,
                "execution_id": event.execution_id,
            })
        timeline.sort(key=lambda x: x["time_offset_ms"])
        return timeline

    def rebuild_graph_events(
        self,
        events: list[RuntimeEvent],
    ) -> dict[str, list[dict[str, Any]]]:
        graph_events: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            if event.event_type in (
                EventType.GRAPH_NODE_STARTED,
                EventType.GRAPH_NODE_COMPLETED,
                EventType.GRAPH_STARTED,
                EventType.GRAPH_COMPLETED,
            ):
                node_id = event.payload.get("node_id", "unknown")
                graph_events.setdefault(node_id, []).append({
                    "type": event.type_name,
                    "timestamp": event.timestamp,
                    "duration_ms": event.duration_ms,
                    "error": event.error,
                })
        return graph_events


trace_rebuilder = TraceRebuilder()
