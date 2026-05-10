from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

from sqlalchemy.orm import Session as SASession

from runtime.security.models import ApiKey as ApiKeyModel


API_KEY_PREFIX = "ak_aiih_"


def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_prefix, key_hash).

    raw_key:  shown to user once (e.g. ak_aiih_a1b2c3d4...)
    key_prefix: first 20 chars for display in list
    key_hash:   SHA-256 hash stored in DB
    """
    raw_key = API_KEY_PREFIX + secrets.token_hex(16)
    key_prefix = raw_key[:20]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def validate_api_key(db: SASession, raw_key: str) -> ApiKeyModel | None:
    key_hash = hash_api_key(raw_key)
    key = (
        db.query(ApiKeyModel)
        .filter(
            ApiKeyModel.key_hash == key_hash,
            ApiKeyModel.is_active == True,
        )
        .first()
    )
    if key is None:
        return None
    if key.expires_at and time.time() > key.expires_at:
        return None
    key.last_used_at = time.time()
    db.commit()
    return key


def create_api_key(
    db: SASession,
    user_id: int,
    name: str = "",
    expires_at: float | None = None,
) -> tuple[ApiKeyModel, str]:
    raw_key, key_prefix, key_hash = generate_api_key()
    key = ApiKeyModel(
        user_id=user_id,
        key_prefix=key_prefix,
        key_hash=key_hash,
        name=name,
        expires_at=expires_at,
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    return key, raw_key


def revoke_api_key(db: SASession, key_id: int, user_id: int | None = None) -> bool:
    query = db.query(ApiKeyModel).filter(ApiKeyModel.id == key_id)
    if user_id is not None:
        query = query.filter(ApiKeyModel.user_id == user_id)
    key = query.first()
    if key is None:
        return False
    key.is_active = False
    db.commit()
    return True


def list_api_keys(db: SASession, user_id: int | None = None) -> list[dict[str, Any]]:
    query = db.query(ApiKeyModel)
    if user_id is not None:
        query = query.filter(ApiKeyModel.user_id == user_id)
    return [k.to_dict() for k in query.order_by(ApiKeyModel.created_at.desc()).all()]
