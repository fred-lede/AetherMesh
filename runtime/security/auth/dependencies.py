from __future__ import annotations

import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session as SASession

from runtime.security.database import get_db
from runtime.security.models import User
from runtime.security.auth.jwt import decode_token


def get_current_user(
    request: Request,
    db: SASession = Depends(get_db),
) -> User:
    token = request.cookies.get("aiih_access_token", "")
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


def require_role(role: str):
    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role != role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return checker


def optional_current_user(
    request: Request,
    db: SASession = Depends(get_db),
) -> User | None:
    try:
        return get_current_user(request, db)
    except HTTPException:
        return None
