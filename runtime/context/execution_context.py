from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from runtime.context.provider_context import ProviderState
from runtime.context.tool_context import ToolState
from runtime.context.gpu_context import GPUState
from runtime.context.session_context import SessionState
from runtime.context.stream_context import StreamState
from runtime.context.memory_context import MemoryState
from runtime.context.security_context import SecurityScope
from runtime.state.execution_state import RuntimeStatus
from runtime.state.graph_state import GraphState
from runtime.state.trace_state import TraceState


@dataclass
class ExecutionContext:
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    runtime_status: RuntimeStatus = RuntimeStatus.CREATED

    provider_state: ProviderState = field(default_factory=ProviderState)
    memory_state: MemoryState = field(default_factory=MemoryState)
    gpu_state: GPUState = field(default_factory=GPUState)
    tool_state: ToolState = field(default_factory=ToolState)
    security_scope: SecurityScope = field(default_factory=SecurityScope)
    stream_state: StreamState = field(default_factory=StreamState)
    graph_state: GraphState = field(default_factory=GraphState)
    trace_state: TraceState = field(default_factory=TraceState)
    session_state: SessionState = field(default_factory=SessionState)

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error: str) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.error = error

    def elapsed_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() * 1000

    def snapshot(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "runtime_status": self.runtime_status.value,
            "provider": self.provider_state.selected_provider,
            "model": self.provider_state.selected_model,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_ms": self.elapsed_ms(),
            "error": self.error,
            "graph_nodes": len(self.graph_state.node_states),
            "tools_called": len(self.tool_state.results),
            "gpu_devices": list(self.gpu_state.allocations.keys()),
            "stream_active": self.stream_state.active,
            "trace_spans": len(self.trace_state.spans),
        }
