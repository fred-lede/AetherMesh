from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from runtime.security import rate_limiter, input_validator
from runtime.security.auth.api_key import validate_api_key
from runtime.security.database import SessionLocal
from runtime.security.models import User

logger = logging.getLogger("security.middleware")


def _verify_api_key(key: str) -> dict | None:
    """Returns {user_id, api_key_id} for valid keys, or None."""
    env_keys = os.getenv("AIIH_API_KEY", "").strip()
    if env_keys:
        env_list = [k.strip() for k in env_keys.split(",") if k.strip()]
        if key in env_list:
            admin_email = os.getenv("AIIH_ADMIN_EMAIL", "").strip().lower()
            if admin_email:
                try:
                    db = SessionLocal()
                    try:
                        admin = db.query(User).filter(User.email == admin_email).first()
                        if admin:
                            return {"user_id": admin.id, "api_key_id": None}
                    finally:
                        db.close()
                except Exception:
                    pass
            return {"user_id": None, "api_key_id": None}
    try:
        db = SessionLocal()
        try:
            key_record = validate_api_key(db, key)
            if key_record is not None:
                return {"user_id": key_record.user_id, "api_key_id": key_record.id}
            return None
        finally:
            db.close()
    except Exception:
        return None


AUTH_BYPASS_PATHS: frozenset[str] = frozenset({
    "/health",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/.well-known/ai-plugin.json",
    "/v1/audio/transcriptions/stream",
})

AUTH_BYPASS_PREFIXES: tuple[str, ...] = (
    "/api/metrics/",
)


def add_security_middleware(
    app: FastAPI,
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    enable_validation: bool = True,
    auth_bypass_paths: set[str] | None = None,
) -> None:
    bypass_paths = set(AUTH_BYPASS_PATHS) | (auth_bypass_paths or set())

    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Any) -> Any:
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key", "") or request.headers.get("authorization", "").replace("Bearer ", "")

        if enable_auth:
            if path in bypass_paths or path.startswith(AUTH_BYPASS_PREFIXES):
                pass
            elif not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"type": "authentication_error", "message": "Missing API key"}},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                key_info = _verify_api_key(api_key)
                if not key_info:
                    return JSONResponse(
                        status_code=401,
                        content={"error": {"type": "authentication_error", "message": "Invalid API key"}},
                    )
                request.state.api_key_id = key_info["api_key_id"]
                request.state.user_id = key_info["user_id"]

        if enable_rate_limit:
            rate_key = api_key or client_ip
            if not rate_limiter.check(rate_key):
                return JSONResponse(
                    status_code=429,
                    content={"error": {"type": "rate_limit_error", "message": "Too many requests"}},
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
