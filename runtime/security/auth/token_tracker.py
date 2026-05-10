from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session as SASession

from runtime.security.models import TokenUsage


def record_token_usage(
    db: SASession,
    user_id: int,
    input_tokens: int,
    output_tokens: int,
    provider: str = "",
    model: str = "",
    api_key_id: int | None = None,
) -> TokenUsage:
    record = TokenUsage(
        user_id=user_id,
        api_key_id=api_key_id,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_token_usage(
    db: SASession,
    user_id: int | None = None,
    from_ts: float | None = None,
    to_ts: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    query = db.query(TokenUsage)
    if user_id is not None:
        query = query.filter(TokenUsage.user_id == user_id)
    if from_ts is not None:
        query = query.filter(TokenUsage.created_at >= from_ts)
    if to_ts is not None:
        query = query.filter(TokenUsage.created_at <= to_ts)
    return [
        r.to_dict()
        for r in query.order_by(TokenUsage.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    ]


def get_token_usage_summary(
    db: SASession,
    user_id: int | None = None,
    from_ts: float | None = None,
    to_ts: float | None = None,
) -> dict[str, Any]:
    records = get_token_usage(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts, limit=10000)
    total_input = sum(r["input_tokens"] for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "record_count": len(records),
    }
