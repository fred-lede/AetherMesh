from __future__ import annotations

import router.traces_router as traces_module
from fastapi import FastAPI
from fastapi.testclient import TestClient

from runtime.observability.execution_trace import execution_trace_collector
from runtime.observability.tracing import tracer
from runtime.security.middleware import add_security_middleware


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(traces_module.router)
    add_security_middleware(app, enable_rate_limit=False)
    return TestClient(app)


def _cleanup() -> None:
    tracer.clear()


def test_traces_empty(monkeypatch) -> None:
    _cleanup()
    monkeypatch.setattr(traces_module, "execution_trace_collector", _FakeCollector())
    client = _make_client()
    response = client.get("/v1/traces")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "traces"
    assert data["span_count"] == 0
    assert data["spans"] == []


def test_traces_lists_seeded_spans(monkeypatch) -> None:
    _cleanup()
    monkeypatch.setattr(traces_module, "execution_trace_collector", _FakeCollector())
    ctx = tracer.start_trace("request")
    tracer.start_span("db.query", parent=ctx)
    tracer.end_span(ctx)
    client = _make_client()
    response = client.get("/v1/traces")
    assert response.status_code == 200
    data = response.json()
    assert data["span_count"] == 2
    assert ctx.trace_id in data["trace_ids"]
    names = {span["name"] for span in data["spans"]}
    assert names == {"request", "db.query"}


def test_traces_filter_by_trace_id() -> None:
    _cleanup()
    ctx_a = tracer.start_trace("request")
    tracer.end_span(ctx_a)
    ctx_b = tracer.start_trace("other")
    tracer.end_span(ctx_b)
    client = _make_client()
    response = client.get("/v1/traces", params={"trace_id": ctx_a.trace_id})
    assert response.status_code == 200
    data = response.json()
    assert len(data["spans"]) == 1
    assert data["spans"][0]["trace_id"] == ctx_a.trace_id


def test_traces_include_execution_summaries(monkeypatch) -> None:
    _cleanup()
    monkeypatch.setattr(
        traces_module,
        "execution_trace_collector",
        _FakeCollector({"exec-1": {"execution_id": "exec-1", "total_duration_ms": 12.5, "span_count": 3}}),
    )
    client = _make_client()
    response = client.get("/v1/traces")
    assert response.status_code == 200
    data = response.json()
    assert len(data["execution_traces"]) == 1
    assert data["execution_traces"][0]["execution_id"] == "exec-1"
    assert data["execution_traces"][0]["span_count"] == 3


def test_export_json() -> None:
    _cleanup()
    ctx = tracer.start_trace("request")
    tracer.end_span(ctx)
    client = _make_client()
    response = client.get("/v1/traces/export", params={"format": "json"})
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "traces"
    assert len(data["spans"]) == 1


def test_export_otlp_payload_shape() -> None:
    _cleanup()
    ctx = tracer.start_trace("request")
    tracer.end_span(ctx)
    client = _make_client()
    response = client.get("/v1/traces/export", params={"format": "otlp", "service_name": "svc-test"})
    assert response.status_code == 200
    data = response.json()
    span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert len(span["traceId"]) == 32
    assert len(span["spanId"]) == 16
    assert span["name"] == "request"
    assert any(a["key"] == "service.name" for a in span["attributes"])


def test_export_invalid_format() -> None:
    client = _make_client()
    response = client.get("/v1/traces/export", params={"format": "bogus"})
    assert response.status_code == 400


def test_export_post_requires_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(traces_module.settings, "otel_endpoint", "")
    monkeypatch.setattr(traces_module.settings, "otel_export_enabled", False)
    client = _make_client()
    response = client.post("/v1/traces/export", json={})
    assert response.status_code == 400


def test_export_post_dispatches(monkeypatch) -> None:
    monkeypatch.setattr(traces_module.settings, "otel_export_enabled", True)
    monkeypatch.setattr(
        traces_module,
        "export_to_collector",
        lambda endpoint=None: {"exported": True, "endpoint": endpoint, "span_count": 0},
    )
    client = _make_client()
    response = client.post("/v1/traces/export", json={"endpoint": "http://collector:4318"})
    assert response.status_code == 200
    data = response.json()
    assert data["exported"] is True
    assert data["endpoint"] == "http://collector:4318"


def test_delete_clears() -> None:
    _cleanup()
    ctx = tracer.start_trace("request")
    tracer.end_span(ctx)
    client = _make_client()
    response = client.delete("/v1/traces")
    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert len(tracer.get_spans()) == 0


def test_request_middleware_seeds_execution_trace() -> None:
    from router.openai_router import app

    tracer.clear()
    execution_trace_collector.clear()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401
    traces = execution_trace_collector.list_traces()
    assert len(traces) == 1
    summary = execution_trace_collector.trace_summary(traces[0])
    assert summary["total_duration_ms"] >= 0.0
    spans = [s for s in tracer.get_spans() if s["name"] == "execution"]
    assert len(spans) == 1
    assert spans[0]["attributes"]["execution_id"] == traces[0]
    tracer.clear()
    execution_trace_collector.clear()


class _FakeCollector:
    def __init__(self, summaries: dict[str, dict] | None = None) -> None:
        self._summaries = summaries or {}

    def list_traces(self) -> list[str]:
        return list(self._summaries)

    def trace_summary(self, execution_id: str) -> dict:
        return self._summaries[execution_id]

    def clear(self) -> None:
        self._summaries.clear()
