from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from runtime.security import api_key_auth, rate_limiter, input_validator

logger = logging.getLogger("security.middleware")


def add_security_middleware(
    app: FastAPI,
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    enable_validation: bool = True,
) -> None:
    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Any) -> Any:
        if request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key", "") or request.headers.get("authorization", "").replace("Bearer ", "")

        if enable_auth and api_key_auth.enabled:
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"type": "error", "error": {"type": "authentication_error", "message": "Missing API key"}},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if not api_key_auth.validate(api_key):
                return JSONResponse(
                    status_code=401,
                    content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}},
                )

        if enable_rate_limit:
            rate_key = api_key or client_ip
            if not rate_limiter.check(rate_key):
                return JSONResponse(
                    status_code=429,
                    content={"type": "error", "error": {"type": "rate_limit_error", "message": "Too many requests"}},
                    headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
                )

        response = await call_next(request)

        if enable_rate_limit:
            rate_key = api_key or client_ip
            remaining = rate_limiter.get_remaining(rate_key)
            response.headers["X-RateLimit-Remaining"] = str(int(remaining))
            response.headers["X-RateLimit-Limit"] = str(int(rate_limiter.default_burst))

        return response


def validate_request_body(body: dict[str, Any]) -> None:
    if "messages" in body:
        input_validator.validate_messages(body["messages"])
    if "prompt" in body:
        body["prompt"] = input_validator.validate_text(body["prompt"], "prompt")
    if "max_tokens" in body:
        if not isinstance(body["max_tokens"], int) or body["max_tokens"] < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")
