from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config.settings import settings
from runtime.orchestration.openai_handler import RouterService
from router.responses_router import create_responses_router
from router.streaming_router import stream_response


service = RouterService()
app = FastAPI(title="AetherMesh Router", version="4.0.0")

if settings.rate_limit_enabled:
    from router.rate_limiter import rate_limit_middleware
    app.middleware("http")(rate_limit_middleware)

app.include_router(create_responses_router(service))


def _error_type_for_status(status_code: int) -> str:
    if status_code == 429:
        return "rate_limit_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "server_error"


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = int(exc.status_code)
    detail = exc.detail
    headers = dict(exc.headers or {})

    code = "internal_error"
    message = "Request failed."
    extra: dict[str, Any] = {}

    if isinstance(detail, dict):
        code = str(detail.get("code") or code)
        message = str(detail.get("message") or message)
        if "retry_after" in detail:
            extra["retry_after"] = detail["retry_after"]
        if "param" in detail:
            extra["param"] = detail["param"]
    elif isinstance(detail, str):
        message = detail
    elif detail is not None:
        message = str(detail)

    error_payload: dict[str, Any] = {
        "message": message,
        "type": _error_type_for_status(status_code),
        "code": code,
    }
    error_payload.update(extra)

    body: dict[str, Any] = {"error": error_payload}
    if request.url.path.startswith("/cluster/"):
        body = {"detail": {"code": code, "message": message, **extra}}

    return JSONResponse(status_code=status_code, content=body, headers=headers)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "router"}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any] = Body(...)):
    if payload.get("stream"):
        return stream_response(service.handle_streaming_chat(payload))
    return service.handle_chat(payload)


@app.post("/v1/rerank")
def rerank(payload: dict[str, Any] = Body(...)):
    return service.handle_rerank(payload)
