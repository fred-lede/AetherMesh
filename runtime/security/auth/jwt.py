from __future__ import annotations

import os
import time
from typing import Any

import jwt

SECRET_KEY_ENV = "AIIH_JWT_SECRET"


def _get_secret() -> str:
    key = os.getenv(SECRET_KEY_ENV, "").strip()
    if not key:
        key = "aiih-dev-secret-do-not-use-in-production"
    return key


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_S = 900  # 15 min
REFRESH_TOKEN_EXPIRE_S = 2592000  # 30 days


def create_access_token(user_id: int, role: str, expires_s: int = ACCESS_TOKEN_EXPIRE_S) -> str:
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_s,
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def create_refresh_token(user_id: int, expires_s: int = REFRESH_TOKEN_EXPIRE_S) -> str:
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_s,
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
