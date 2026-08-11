from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import settings
from runtime.observability.tracing import tracer

logger = logging.getLogger("runtime.observability.otel_exporter")


def _pad_hex(value: str, length: int) -> str:
    return value.lstrip("0").rjust(length, "0") if value else "0" * length


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, (int, float)):
        v = {"doubleValue": float(value)} if isinstance(value, float) else {"intValue": int(value)}
    elif isinstance(value, dict):
        v = {"stringValue": str(value)}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _span_to_otlp(span: Any, service_name: str = "aethermesh") -> dict[str, Any]:
    start = getattr(span, "start_time", 0.0) or 0.0
    end = getattr(span, "end_time", 0.0) or start
    attributes = [{"key": "service.name", "value": {"stringValue": service_name}}]
    for key, value in (span.attributes or {}).items():
        attributes.append(_attr(str(key), value))
    return {
        "traceId": _pad_hex(span.trace_id, 32),
        "spanId": _pad_hex(span.span_id, 16),
        "parentSpanId": _pad_hex(span.parent_span_id, 16) if span.parent_span_id else "",
        "name": span.name,
        "kind": 1,
        "startTimeUnixNano": str(int(start * 1_000_000_000)),
        "endTimeUnixNano": str(int(end * 1_000_000_000)),
        "attributes": attributes,
    }


def collect_tracer_spans() -> list[Any]:
    return list(tracer._spans)


def build_otlp_payload(service_name: str = "aethermesh") -> dict[str, Any]:
    spans = [_span_to_otlp(s, service_name) for s in collect_tracer_spans()]
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
                        {"key": "telemetry.sdk.name", "value": {"stringValue": "aethermesh"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": service_name},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def export_to_collector(endpoint: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    endpoint = endpoint or settings.otel_endpoint
    if not endpoint:
        return {"exported": False, "error": "no OTEL endpoint configured"}
    payload = build_otlp_payload()
    span_count = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])
    try:
        response = requests.post(
            endpoint.rstrip("/") + "/v1/traces",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("OTLP export to %s failed: %s", endpoint, exc)
        return {"exported": False, "error": str(exc), "span_count": span_count}
    return {"exported": True, "endpoint": endpoint, "span_count": span_count}


def now_iso() -> float:
    return time.time()
