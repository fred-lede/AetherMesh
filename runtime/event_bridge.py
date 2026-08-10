from __future__ import annotations

import logging
from typing import Any

from runtime.events.bus import runtime_event_bus
from runtime.events.event import RuntimeEvent, event_from_type
from runtime.events.event_types import EventType
from runtime.observability.event_bus import graph_event_bus, GraphEvent

logger = logging.getLogger("bridge.event_bus")

_BRIDGE_MARKER = "_aether_bridge_forwarded"


def _graph_event_to_runtime_event(graph_event: GraphEvent) -> RuntimeEvent:
    type_mapping: dict[str, EventType] = {
        "node_started": EventType.GRAPH_NODE_STARTED,
        "node_completed": EventType.GRAPH_NODE_COMPLETED,
        "node_failed": EventType.GRAPH_NODE_COMPLETED,
        "graph_started": EventType.GRAPH_STARTED,
        "graph_completed": EventType.GRAPH_COMPLETED,
    }
    event_type = type_mapping.get(graph_event.type, EventType.GRAPH_NODE_STARTED)
    return RuntimeEvent(
        event_type=event_type,
        execution_id=graph_event.trace_id,
        source="graph_executor",
        duration_ms=graph_event.duration_ms,
        error=graph_event.error,
        payload={
            "node_id": graph_event.node_id,
            "node_type": graph_event.node_type,
            "span_id": graph_event.span_id,
            "content": graph_event.content,
            **(graph_event.metadata or {}),
        },
    )


def _runtime_event_to_graph_event(event: RuntimeEvent) -> GraphEvent | None:
    reverse_mapping: dict[EventType, str] = {
        EventType.GRAPH_NODE_STARTED: "node_started",
        EventType.GRAPH_NODE_COMPLETED: "node_completed",
        EventType.GRAPH_STARTED: "graph_started",
        EventType.GRAPH_COMPLETED: "graph_completed",
    }
    graph_type = reverse_mapping.get(event.event_type)
    if not graph_type:
        return None
    return GraphEvent(
        type=graph_type,
        node_id=event.payload.get("node_id", ""),
        node_type=event.payload.get("node_type", ""),
        trace_id=event.execution_id,
        span_id=event.payload.get("span_id", ""),
        timestamp=event.timestamp,
        duration_ms=event.duration_ms,
        error=event.error,
        content=event.payload.get("content"),
        metadata={**event.payload, _BRIDGE_MARKER: True},
    )


def _old_bus_to_new_bridge(graph_event: GraphEvent) -> None:
    if graph_event.metadata.get(_BRIDGE_MARKER):
        return
    runtime_event = _graph_event_to_runtime_event(graph_event)
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            asyncio.ensure_future(runtime_event_bus.publish(runtime_event))
        else:
            runtime_event_bus.publish_sync(runtime_event)
    except RuntimeError:
        runtime_event_bus.publish_sync(runtime_event)


def _new_bus_to_old_bridge(event: RuntimeEvent) -> None:
    graph_event = _runtime_event_to_graph_event(event)
    if graph_event is not None:
        graph_event_bus.emit(graph_event)


def connect_event_buses() -> None:
    graph_event_bus.subscribe(_old_bus_to_new_bridge)
    runtime_event_bus.subscribe(_new_bus_to_old_bridge)
    logger.info("Event bus bridge connected: graph_event_bus <-> runtime_event_bus")
