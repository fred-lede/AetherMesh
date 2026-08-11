from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from runtime.security.auth.token_tracker import (
    get_token_usage,
    get_token_usage_breakdown,
    get_token_usage_summary,
)
from runtime.security.database import SessionLocal

router = APIRouter(prefix="/v1/usage")


def _query_rows(
    from_ts: float | None,
    to_ts: float | None,
    model: str | None,
    provider: str | None,
    limit: int,
    offset: int,
    user_id: int | None = None,
) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        return get_token_usage(
            db,
            user_id=user_id,
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
            offset=offset,
            model=model,
            provider=provider,
        )
    finally:
        db.close()


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "timestamp", "user_id", "api_key_id", "provider", "model", "input_tokens", "output_tokens", "total_tokens"])
    for row in rows:
        writer.writerow([
            row.get("id", ""),
            row.get("created_at", ""),
            row.get("user_id", ""),
            row.get("api_key_id", ""),
            row.get("provider", ""),
            row.get("model", ""),
            row.get("input_tokens", 0),
            row.get("output_tokens", 0),
            row.get("total_tokens", 0),
        ])
    return buffer.getvalue()


@router.get("/export")
def export_usage(
    format: str = Query(default="json", pattern="^(csv|json)$"),
    from_ts: float | None = None,
    to_ts: float | None = None,
    model: str | None = None,
    provider: str | None = None,
    limit: int = Query(default=1000, ge=1, le=100000),
    offset: int = Query(default=0, ge=0),
    user_id: int | None = None,
    group_by: str | None = Query(default=None, pattern="^(model|provider)$"),
) -> Any:
    rows = _query_rows(from_ts, to_ts, model, provider, limit, offset, user_id=user_id)
    if format == "csv":
        csv_content = _rows_to_csv(rows)
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="usage_export.csv"'},
        )
    db = SessionLocal()
    try:
        summary = get_token_usage_summary(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts)
        breakdown = None
        if group_by:
            breakdown = get_token_usage_breakdown(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts, group_by=group_by)
    finally:
        db.close()
    return {
        "object": "usage_export",
        "format": "json",
        "summary": summary,
        "breakdown": breakdown,
        "count": len(rows),
        "data": rows,
    }


@router.get("/summary")
def usage_summary(
    from_ts: float | None = None,
    to_ts: float | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        summary = get_token_usage_summary(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts)
        by_model = get_token_usage_breakdown(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts, group_by="model")
        by_provider = get_token_usage_breakdown(db, user_id=user_id, from_ts=from_ts, to_ts=to_ts, group_by="provider")
    finally:
        db.close()
    return {"object": "usage_summary", "summary": summary, "by_model": by_model, "by_provider": by_provider}
