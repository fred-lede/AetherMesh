from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any

_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar("trace_context", default=None)

MAX_SPANS = 1000


@dataclass
class Span:
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    trace_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class TraceContext:
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        tracer.end_span(self)

    def set_attribute(self, key: str, value: Any) -> None:
        for span in tracer._spans:
            if span.span_id == self.span_id:
                span.attributes[key] = value
                return


class Tracer:
    def __init__(self, max_spans: int = MAX_SPANS) -> None:
        self._max_spans = max_spans
        self._spans: list[Span] = []

    def _append_span(self, span: Span) -> None:
        self._spans.append(span)
        if len(self._spans) > self._max_spans:
            del self._spans[: len(self._spans) - self._max_spans]

    def start_trace(self, name: str = "request") -> TraceContext:
        trace_id = uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:16]
        ctx = TraceContext(trace_id=trace_id, span_id=span_id)
        _current_trace.set(ctx)
        self._append_span(Span(
            span_id=span_id,
            name=name,
            trace_id=trace_id,
            start_time=__import__("time").time(),
        ))
        return ctx

    def start_span(
        self,
        name: str,
        parent: TraceContext | None = None,
    ) -> TraceContext:
        parent_ctx = parent or _current_trace.get()
        trace_id = parent_ctx.trace_id if parent_ctx else uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = parent_ctx.span_id if parent_ctx else ""
        ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        _current_trace.set(ctx)
        self._append_span(Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            trace_id=trace_id,
            start_time=__import__("time").time(),
        ))
        return ctx

    def end_span(self, ctx: TraceContext | None = None) -> None:
        target = ctx or _current_trace.get()
        if not target:
            return
        now = __import__("time").time()
        for span in reversed(self._spans):
            if span.span_id == target.span_id and span.end_time == 0.0:
                span.end_time = now
                break

    def get_spans(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        spans = self._spans
        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        return [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "trace_id": s.trace_id,
                "elapsed_ms": s.elapsed_ms,
                "attributes": s.attributes,
            }
            for s in spans
        ]

    def clear(self) -> None:
        self._spans.clear()
        _current_trace.set(None)


tracer = Tracer()
