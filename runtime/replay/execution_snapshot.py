from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.bus import runtime_event_bus
from runtime.events.event_types import EventType


@dataclass
class ExecutionSnapshot:
    execution_id: str = ""
    timestamp: float = 0.0
    runtime_status: str = ""
    provider: str = ""
    model: str = ""
    graph_nodes: dict[str, str] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    gpu_allocations: dict[str, Any] = field(default_factory=dict)
    stream_active: bool = False
    error: str = ""
    elapsed_ms: float = 0.0
    event_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "runtime_status": self.runtime_status,
            "provider": self.provider,
            "model": self.model,
            "graph_nodes": self.graph_nodes,
            "tool_results": self.tool_results,
            "gpu_allocations": self.gpu_allocations,
            "stream_active": self.stream_active,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "event_count": self.event_count,
            "metadata": self.metadata,
        }
