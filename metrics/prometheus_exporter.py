from __future__ import annotations

from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Response

from config.settings import settings
from metrics.metrics import render_prometheus_text


app = FastAPI(title="AI Inference Hub Prometheus Exporter", version="4.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "metrics-exporter"}


@app.get("/metrics")
def metrics() -> Response:
    try:
        response = requests.get(f"{settings.control_plane_url}/cluster/metrics", timeout=5)
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Control plane unavailable: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    payload = render_prometheus_text(response.json())
    return Response(content=payload, media_type="text/plain; version=0.0.4; charset=utf-8")
