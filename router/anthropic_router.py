from __future__ import annotations

"""Anthropic-compatible API router. Thin wrapper that delegates to runtime/."""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config.settings import settings
from metrics.request_metrics import RequestRecord, request_metrics
from providers.base import ProviderError
from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.capabilities import required_anthropic_capabilities
from runtime.orchestration.routing_engine import routing_engine
from runtime.orchestration.streaming import stream_anthropic_with_metrics
from runtime.security.tool_policy import evaluate_server_tool_policy
from runtime.security.middleware import add_security_middleware
from runtime.gpu_os.routes import gpu_router
from runtime.multi_agent.routes import agent_router
from runtime.tools.builtin.web_search import stream_web_server_tool_response
from router.anthropic.messages_adapter import create_messages_routes

anthropic_service = AnthropicRouter()
logger = logging.getLogger("anthropic_router")


class ASCIISafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")


ANTHROPIC_RATE_LIMIT_HEADERS = [
    "request-id", "retry-after",
    "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset", "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining", "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-input-tokens-reset", "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-output-tokens-remaining", "anthropic-ratelimit-output-tokens-reset",
    "anthropic-ratelimit-retry-after", "anthropic-ratelimit-tier", "cf-ray",
]


app = FastAPI(title="AetherMesh - Anthropic Compatible", version="4.0.0")

add_security_middleware(app)

app.include_router(gpu_router)
app.include_router(agent_router)


@app.get("/api/metrics/requests")
def metrics_requests() -> dict[str, Any]:
    return request_metrics.get_summary()


@app.get("/api/metrics/providers")
def metrics_providers() -> dict[str, Any]:
    return {"providers": request_metrics.get_provider_metrics()}


@app.get("/api/metrics/provider-diagnostics")
def metrics_provider_diagnostics() -> dict[str, Any]:
    return {"providers": request_metrics.get_provider_diagnostics()}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "anthropic_router"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return anthropic_service.list_models()


create_messages_routes(app, anthropic_service)


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = int(exc.status_code)
    detail = exc.detail

    error_type = "api_error"
    if isinstance(detail, dict):
        error_type = detail.get("type", error_type)

    if status_code == 429:
        error_type = "rate_limit_error"
    elif status_code == 400:
        error_type = detail.get("type", "invalid_request_error") if isinstance(detail, dict) else "invalid_request_error"
    elif status_code in (502, 503, 504):
        error_type = "overloaded_error" if status_code == 503 else "api_error"

    if isinstance(detail, dict):
        error_payload = {"type": error_type, "message": detail.get("message", "Request failed.")}
    elif isinstance(detail, str):
        error_payload = {"type": error_type, "message": detail}
    else:
        error_payload = {"type": error_type, "message": "Request failed."}

    headers: dict[str, str] = dict(exc.headers or {})
    headers["X-Request-Id"] = f"req_{uuid.uuid4().hex[:24]}"
    if status_code == 429:
        headers.setdefault("Retry-After", "60")

    return ASCIISafeJSONResponse(
        status_code=status_code,
        content={"type": "error", "error": error_payload},
        media_type="application/json; charset=utf-8",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
