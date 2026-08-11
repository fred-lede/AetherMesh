from __future__ import annotations

from runtime.context.execution_context import ExecutionContext
from runtime.observability.execution_trace import execution_trace_collector
from runtime.observability.tracing import TraceContext, tracer


def _cleanup() -> None:
    tracer.clear()
    execution_trace_collector.clear()


def test_trace_context_is_context_manager() -> None:
    _cleanup()
    ctx = tracer.start_trace("request")
    with ctx:
        assert isinstance(ctx, TraceContext)
    spans = tracer.get_spans()
    assert spans[0]["elapsed_ms"] >= 0.0


def test_trace_context_set_attribute() -> None:
    _cleanup()
    ctx = tracer.start_trace("request")
    ctx.set_attribute("execution_id", "exec-123")
    ctx.set_attribute("session_id", "ses-456")
    spans = tracer.get_spans()
    assert spans[0]["attributes"]["execution_id"] == "exec-123"
    assert spans[0]["attributes"]["session_id"] == "ses-456"


def test_start_trace_seeds_tracer_span() -> None:
    _cleanup()
    ctx = ExecutionContext(session_id="ses-abc")
    execution_trace_collector.start_trace(ctx)
    spans = tracer.get_spans()
    assert len(spans) == 1
    assert spans[0]["name"] == "execution"
    assert spans[0]["attributes"]["execution_id"] == ctx.execution_id
    assert spans[0]["attributes"]["session_id"] == "ses-abc"


def test_end_trace_closes_span_with_duration() -> None:
    _cleanup()
    ctx = ExecutionContext()
    execution_trace_collector.start_trace(ctx)
    execution_trace_collector.end_trace(ctx)
    summary = execution_trace_collector.trace_summary(ctx.execution_id)
    assert summary["total_duration_ms"] >= 0.0
    spans = tracer.get_spans()
    assert len(spans) == 1
    assert spans[0]["elapsed_ms"] >= 0.0


def test_end_trace_without_start_is_noop() -> None:
    _cleanup()
    execution_trace_collector.end_trace(ExecutionContext())
    assert tracer.get_spans() == []


def test_clear_resets_tracer_bridge() -> None:
    _cleanup()
    ctx = ExecutionContext()
    execution_trace_collector.start_trace(ctx)
    execution_trace_collector.clear()
    assert execution_trace_collector.list_traces() == []
    assert tracer.get_spans() == []


def test_tracer_caps_spans() -> None:
    from runtime.observability.tracing import Tracer

    _cleanup()
    small = Tracer(max_spans=5)
    ctx = None
    for _ in range(7):
        ctx = small.start_trace("request")
        small.end_span(ctx)
    spans = small.get_spans()
    assert len(spans) == 5
    assert ctx is not None
    assert spans[-1]["span_id"] == ctx.span_id


def test_collector_caps_traces() -> None:
    from runtime.observability.execution_trace import ExecutionTraceCollector

    _cleanup()
    small = ExecutionTraceCollector(max_traces=3)
    ctxs = [ExecutionContext() for _ in range(5)]
    for ctx in ctxs:
        small.start_trace(ctx)
    remaining = small.list_traces()
    assert len(remaining) == 3
    assert ctxs[0].execution_id not in remaining
    assert ctxs[4].execution_id in remaining
    evicted_span = [s for s in tracer.get_spans() if s["attributes"].get("execution_id") == ctxs[0].execution_id]
    assert len(evicted_span) == 1
    assert evicted_span[0]["elapsed_ms"] >= 0.0
