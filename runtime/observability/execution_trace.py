from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.context.execution_context import ExecutionContext
from runtime.observability.tracing import tracer


@dataclass
class ExecutionTraceSpan:
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class ExecutionTraceRecord:
    execution_id: str = ""
    spans: list[ExecutionTraceSpan] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class ExecutionTraceCollector:
    def __init__(self) -> None:
        self._traces: dict[str, ExecutionTraceRecord] = {}

    def start_trace(self, ctx: ExecutionContext) -> None:
        record = ExecutionTraceRecord(
            execution_id=ctx.execution_id,
            start_time=time.time(),
        )
        self._traces[ctx.execution_id] = record
        with tracer.start_span("execution") as span:
            span.set_attribute("execution_id", ctx.execution_id)
            span.set_attribute("session_id", ctx.session_id)

    def end_trace(self, ctx: ExecutionContext) -> None:
        record = self._traces.get(ctx.execution_id)
        if record:
            record.end_time = time.time()

    def add_span(
        self,
        execution_id: str,
        name: str,
        attributes: dict[str, str] | None = None,
    ) -> ExecutionTraceSpan:
        record = self._traces.setdefault(
            execution_id,
            ExecutionTraceRecord(execution_id=execution_id, start_time=time.time()),
        )
        span = ExecutionTraceSpan(
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        record.spans.append(span)
        return span

    def close_span(self, span: ExecutionTraceSpan) -> None:
        span.end_time = time.time()

    def get_trace(self, execution_id: str) -> ExecutionTraceRecord | None:
        return self._traces.get(execution_id)

    def list_traces(self) -> list[str]:
        return list(self._traces.keys())

    def clear(self) -> None:
        self._traces.clear()

    def trace_summary(self, execution_id: str) -> dict[str, Any]:
        record = self._traces.get(execution_id)
        if not record:
            return {}
        return {
            "execution_id": record.execution_id,
            "total_duration_ms": record.total_duration_ms,
            "span_count": len(record.spans),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                }
                for s in record.spans
            ],
        }


execution_trace_collector = ExecutionTraceCollector()
