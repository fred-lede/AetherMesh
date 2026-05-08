from __future__ import annotations

from runtime.observability.event_bus import EventBus, GraphEvent
from runtime.observability.tracing import Tracer
from runtime.observability.metrics import MetricsCollector


def test_event_bus_emit_and_subscribe() -> None:
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(lambda e: received.append(e.type))
    bus.emit(GraphEvent(type="test_event"))
    assert received == ["test_event"]


def test_event_bus_node_started_factory() -> None:
    e = EventBus.node_started(node_id="n1", node_type="llm_call")
    assert e.type == "node_started"
    assert e.node_id == "n1"
    assert e.node_type == "llm_call"


def test_event_bus_node_completed_factory() -> None:
    e = EventBus.node_completed(node_id="n1", duration_ms=100.0, content="result")
    assert e.type == "node_completed"
    assert e.duration_ms == 100.0
    assert e.content == "result"


def test_event_bus_node_failed_factory() -> None:
    e = EventBus.node_failed(node_id="n1", error="boom")
    assert e.type == "node_failed"
    assert e.error == "boom"


def test_event_bus_graph_started_factory() -> None:
    e = EventBus.graph_started(trace_id="trace1")
    assert e.type == "graph_started"
    assert e.trace_id == "trace1"


def test_event_bus_graph_completed_factory() -> None:
    e = EventBus.graph_completed(success=True, elapsed_ms=500.0)
    assert e.type == "graph_completed"
    assert e.content["success"] is True


def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    calls = 0
    def handler(e: GraphEvent) -> None:
        nonlocal calls
        calls += 1
    bus.subscribe(handler)
    bus.emit(GraphEvent(type="e1"))
    bus.unsubscribe(handler)
    bus.emit(GraphEvent(type="e2"))
    assert calls == 1


def test_tracer_start_trace() -> None:
    t = Tracer()
    ctx = t.start_trace("test")
    assert ctx.trace_id != ""
    assert ctx.span_id != ""
    spans = t.get_spans(ctx.trace_id)
    assert len(spans) == 1
    assert spans[0]["name"] == "test"


def test_tracer_start_span() -> None:
    t = Tracer()
    parent = t.start_trace("parent")
    child = t.start_span("child", parent=parent)
    t.end_span(child)
    t.end_span(parent)
    spans = t.get_spans(parent.trace_id)
    assert len(spans) == 2
    names = {s["name"] for s in spans}
    assert names == {"parent", "child"}


def test_tracer_clear() -> None:
    t = Tracer()
    t.start_trace("t")
    t.clear()
    assert t.get_spans() == []


def test_metrics_counter() -> None:
    m = MetricsCollector()
    m.increment("requests")
    m.increment("requests", 3)
    assert m.get_counter("requests") == 4


def test_metrics_histogram() -> None:
    m = MetricsCollector()
    m.record("latency", 10.0)
    m.record("latency", 20.0)
    m.record("latency", 30.0)
    h = m.get_histogram("latency")
    assert h["count"] == 3
    assert h["avg"] == 20.0
    assert h["min"] == 10.0
    assert h["max"] == 30.0


def test_metrics_empty_histogram() -> None:
    m = MetricsCollector()
    h = m.get_histogram("nonexistent")
    assert h["count"] == 0


def test_metrics_gauge() -> None:
    m = MetricsCollector()
    m.set_gauge("temperature", 42.5)
    assert m.get_gauge("temperature") == 42.5


def test_metrics_snapshot() -> None:
    m = MetricsCollector()
    m.increment("req", 5)
    m.record("lat", 15.0)
    m.set_gauge("temp", 30.0)
    s = m.snapshot()
    assert s["counter:req"] == 5
    assert s["histogram:lat"]["avg"] == 15.0
    assert s["gauge:temp"] == 30.0


def test_metrics_clear() -> None:
    m = MetricsCollector()
    m.increment("x", 1)
    m.clear()
    assert m.get_counter("x") == 0
