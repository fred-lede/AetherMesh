from __future__ import annotations

import asyncio
from typing import Any

from runtime.event_bridge import _BRIDGE_MARKER, connect_event_buses
from runtime.events.bus import runtime_event_bus
from runtime.observability.event_bus import EventBus, GraphEvent, graph_event_bus


async def test_event_bridge_no_infinite_loop() -> None:
    graph_seen: list[GraphEvent] = []
    runtime_seen: list[Any] = []

    try:
        graph_event_bus.subscribe(graph_seen.append)
        runtime_event_bus.subscribe(lambda e: runtime_seen.append(e))
        connect_event_buses()

        graph_event_bus.emit(
            EventBus.node_started(node_id="n1", node_type="llm_call", trace_id="t1")
        )

        await asyncio.sleep(0.2)

        assert len(graph_seen) == 2
        assert len(runtime_seen) == 1
        assert runtime_seen[0].payload["node_id"] == "n1"
        echo = graph_seen[1]
        assert echo.metadata.get(_BRIDGE_MARKER) is True
    finally:
        graph_event_bus.clear()
        runtime_event_bus.clear()


async def test_event_bridge_roundtrip_metadata_preserved() -> None:
    runtime_seen: list[Any] = []

    try:
        runtime_event_bus.subscribe(lambda e: runtime_seen.append(e))
        connect_event_buses()

        graph_event_bus.emit(
            EventBus.node_completed(
                node_id="n1", node_type="tool_call", duration_ms=12.0, trace_id="t2"
            )
        )

        await asyncio.sleep(0.2)

        assert len(runtime_seen) == 1
        assert runtime_seen[0].payload["node_id"] == "n1"
        assert runtime_seen[0].payload["node_type"] == "tool_call"
    finally:
        graph_event_bus.clear()
        runtime_event_bus.clear()
