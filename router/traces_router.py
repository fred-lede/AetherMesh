from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from config.settings import settings
from runtime.observability.execution_trace import execution_trace_collector
from runtime.observability.otel_exporter import build_otlp_payload, export_to_collector
from runtime.observability.tracing import tracer

router = APIRouter(prefix="/v1/traces")


def _execution_traces() -> list[dict[str, Any]]:
    summaries = []
    for execution_id in execution_trace_collector.list_traces():
        try:
            summaries.append(execution_trace_collector.trace_summary(execution_id))
        except KeyError:
            continue
    return summaries


@router.get("")
def list_traces(trace_id: str | None = None, limit: int = 500):
    spans = tracer.get_spans(trace_id=trace_id)
    return {
        "object": "traces",
        "span_count": len(spans),
        "trace_ids": sorted({s["trace_id"] for s in spans}),
        "spans": spans[-max(1, min(limit, 10000)):],
        "execution_traces": _execution_traces(),
    }


@router.get("/export")
def export_traces(format: str = "otlp", service_name: str = "aethermesh"):
    if format == "json":
        return {"object": "traces", "spans": tracer.get_spans()}
    if format != "otlp":
        raise HTTPException(status_code=400, detail="format must be 'json' or 'otlp'")
    return JSONResponse(content=build_otlp_payload(service_name=service_name))


@router.post("/export")
def export_now(payload: dict[str, Any] = Body(default={})):
    endpoint = payload.get("endpoint") or settings.otel_endpoint
    if not endpoint and not settings.otel_export_enabled:
        raise HTTPException(status_code=400, detail="No OTEL endpoint configured (set AIIH_OTEL_ENDPOINT or pass 'endpoint')")
    result = export_to_collector(endpoint=endpoint)
    return result


@router.delete("")
def clear_traces():
    tracer.clear()
    execution_trace_collector.clear()
    return {"cleared": True}
