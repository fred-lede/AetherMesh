from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GraphEvent:
    type: str = ""
    node_id: str = ""
    node_type: str = ""
    trace_id: str = ""
    span_id: str = ""
    timestamp: float = 0.0
    duration_ms: float = 0.0
    error: str = ""
    content: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[GraphEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        self._handlers.remove(handler)

    def emit(self, event: GraphEvent) -> None:
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

    def clear(self) -> None:
        self._handlers.clear()

    @staticmethod
    def node_started(
        node_id: str,
        node_type: str = "",
        trace_id: str = "",
        span_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> GraphEvent:
        return GraphEvent(
            type="node_started",
            node_id=node_id,
            node_type=node_type,
            trace_id=trace_id,
            span_id=span_id,
            timestamp=time.time(),
            metadata=metadata or {},
        )

    @staticmethod
    def node_completed(
        node_id: str,
        node_type: str = "",
        duration_ms: float = 0.0,
        trace_id: str = "",
        span_id: str = "",
        content: Any = None,
    ) -> GraphEvent:
        return GraphEvent(
            type="node_completed",
            node_id=node_id,
            node_type=node_type,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
            timestamp=time.time(),
            content=content,
        )

    @staticmethod
    def node_failed(
        node_id: str,
        node_type: str = "",
        error: str = "",
        duration_ms: float = 0.0,
        trace_id: str = "",
        span_id: str = "",
    ) -> GraphEvent:
        return GraphEvent(
            type="node_failed",
            node_id=node_id,
            node_type=node_type,
            error=error,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
            timestamp=time.time(),
        )

    @staticmethod
    def graph_started(
        trace_id: str = "",
        span_id: str = "",
    ) -> GraphEvent:
        return GraphEvent(
            type="graph_started",
            trace_id=trace_id,
            span_id=span_id,
            timestamp=time.time(),
        )

    @staticmethod
    def graph_completed(
        success: bool = True,
        elapsed_ms: float = 0.0,
        trace_id: str = "",
        span_id: str = "",
    ) -> GraphEvent:
        return GraphEvent(
            type="graph_completed",
            timestamp=time.time(),
            error="" if success else "graph_failed",
            duration_ms=elapsed_ms,
            trace_id=trace_id,
            span_id=span_id,
            content={"success": success, "elapsed_ms": elapsed_ms},
        )


graph_event_bus = EventBus()
