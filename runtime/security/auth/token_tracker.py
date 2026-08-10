from __future__ import annotations

import time
from typing import Any

from sqlalchemy import func
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
    query = db.query(
        func.count(TokenUsage.id),
        func.coalesce(func.sum(TokenUsage.input_tokens), 0),
        func.coalesce(func.sum(TokenUsage.output_tokens), 0),
    )
    if user_id is not None:
        query = query.filter(TokenUsage.user_id == user_id)
    if from_ts is not None:
        query = query.filter(TokenUsage.created_at >= from_ts)
    if to_ts is not None:
        query = query.filter(TokenUsage.created_at <= to_ts)
    count, total_input, total_output = query.one()
    total_input = int(total_input)
    total_output = int(total_output)
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "record_count": int(count),
    }


def get_api_key_usage(
    db: SASession,
    api_key_ids: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    query = db.query(
        TokenUsage.api_key_id,
        func.count(TokenUsage.id),
        func.coalesce(func.sum(TokenUsage.input_tokens), 0),
        func.coalesce(func.sum(TokenUsage.output_tokens), 0),
    ).filter(TokenUsage.api_key_id.isnot(None))
    if api_key_ids:
        query = query.filter(TokenUsage.api_key_id.in_(api_key_ids))
    result: dict[int, dict[str, Any]] = {}
    for key_id, count, total_input, total_output in query.group_by(TokenUsage.api_key_id).all():
        total_input = int(total_input)
        total_output = int(total_output)
        result[int(key_id)] = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "record_count": int(count),
        }
    return result
