from __future__ import annotations

import csv
import io
import time

from runtime.security.auth.token_tracker import get_token_usage_breakdown, record_token_usage
from runtime.security.database import SessionLocal, engine
from runtime.security.models import Base, TokenUsage
from router.usage_router import _rows_to_csv


_BASE = time.time()


def _insert_rows():
    global _BASE
    db = SessionLocal()
    try:
        Base.metadata.create_all(engine)
        db.query(TokenUsage).delete()
        db.commit()
        _BASE = time.time()
        for i, (model, provider, inp, out) in enumerate([
            ("qwen3:27b", "ollama", 100, 50),
            ("qwen3:27b", "ollama", 200, 60),
            ("gpt-4o", "openai", 300, 70),
        ]):
            record_token_usage(
                db,
                user_id=1,
                input_tokens=inp,
                output_tokens=out,
                provider=provider,
                model=model,
                api_key_id=2,
            )
            record = db.query(TokenUsage).order_by(TokenUsage.id.desc()).first()
            record.created_at = _BASE + i
            db.commit()
    finally:
        db.close()


def _rows() -> list[dict]:
    db = SessionLocal()
    try:
        rows = db.query(TokenUsage).order_by(TokenUsage.created_at).all()
        return [r.to_dict() for r in rows]
    finally:
        db.close()


def test_rows_to_csv_shape():
    rows = [
        {"id": 1, "created_at": 100.0, "user_id": 1, "api_key_id": 2, "provider": "ollama", "model": "qwen", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    ]
    content = _rows_to_csv(rows)
    parsed = list(csv.reader(io.StringIO(content)))
    assert parsed[0] == ["id", "timestamp", "user_id", "api_key_id", "provider", "model", "input_tokens", "output_tokens", "total_tokens"]
    assert parsed[1][5] == "qwen"


def test_get_token_usage_filters():
    _insert_rows()
    db = SessionLocal()
    try:
        by_model = get_token_usage_breakdown(db, group_by="model")
        assert {b["model"] for b in by_model} == {"qwen3:27b", "gpt-4o"}
        by_provider = get_token_usage_breakdown(db, group_by="provider")
        assert {b["provider"] for b in by_provider} == {"ollama", "openai"}
        ollama = next(b for b in by_provider if b["provider"] == "ollama")
        assert ollama["requests"] == 2
        assert ollama["total_tokens"] == 410
    finally:
        db.close()


def test_get_token_usage_breakdown_time_filter():
    _insert_rows()
    db = SessionLocal()
    try:
        filtered = get_token_usage_breakdown(db, group_by="model", from_ts=_BASE + 1.5, to_ts=_BASE + 2.5)
        assert [b["model"] for b in filtered] == ["gpt-4o"]
    finally:
        db.close()
