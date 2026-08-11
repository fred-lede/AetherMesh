from __future__ import annotations

import os
import tempfile

_db_path = tempfile.mktemp(suffix=".db")
os.environ["AIIH_DB_PATH"] = _db_path

from runtime.security.database import SessionLocal, engine, init_db
from runtime.security.models import ApiKey, TokenUsage, User
from runtime.security.auth.token_tracker import (
    get_api_key_usage,
    get_token_usage_summary,
    get_user_usage,
    record_token_usage,
)

init_db()


def _make_user(db, email: str = "user@test.local") -> User:
    user = User(
        email=email,
        password_hash="x",
        display_name="tester",
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_summary_counts_all_records_without_cap() -> None:
    db = SessionLocal()
    try:
        user = _make_user(db)
        db.add_all(
            TokenUsage(
                user_id=user.id,
                provider="ollama",
                model="m",
                input_tokens=i + 1,
                output_tokens=2,
                total_tokens=i + 3,
            )
            for i in range(10005)
        )
        db.commit()
        summary = get_token_usage_summary(db, user_id=user.id)
        assert summary["record_count"] == 10005
        expected_input = sum(i + 1 for i in range(10005))
        assert summary["total_input_tokens"] == expected_input
        assert summary["total_output_tokens"] == 10005 * 2
        assert summary["total_tokens"] == expected_input + 10005 * 2
    finally:
        db.close()


def test_summary_empty() -> None:
    db = SessionLocal()
    try:
        user = _make_user(db, email="empty@test.local")
        summary = get_token_usage_summary(db, user_id=user.id)
        assert summary["record_count"] == 0
        assert summary["total_tokens"] == 0
    finally:
        db.close()


def test_summary_isolates_users() -> None:
    db = SessionLocal()
    try:
        user_a = _make_user(db, email="a@test.local")
        user_b = _make_user(db, email="b@test.local")
        record_token_usage(db, user_id=user_a.id, input_tokens=10, output_tokens=5)
        record_token_usage(db, user_id=user_a.id, input_tokens=20, output_tokens=6)
        record_token_usage(db, user_id=user_b.id, input_tokens=100, output_tokens=50)
        summary_a = get_token_usage_summary(db, user_id=user_a.id)
        summary_b = get_token_usage_summary(db, user_id=user_b.id)
        assert summary_a["record_count"] == 2
        assert summary_a["total_input_tokens"] == 30
        assert summary_b["record_count"] == 1
        assert summary_b["total_input_tokens"] == 100
    finally:
        db.close()


def test_summary_respects_time_range() -> None:
    db = SessionLocal()
    try:
        user = _make_user(db, email="range@test.local")
        now = user.created_at
        record_token_usage(db, user_id=user.id, input_tokens=5, output_tokens=1)
        record_token_usage(db, user_id=user.id, input_tokens=7, output_tokens=2)
        summary = get_token_usage_summary(db, user_id=user.id, from_ts=now, to_ts=now + 1)
        assert summary["record_count"] == 2
        assert summary["total_input_tokens"] == 12
    finally:
        db.close()


def test_api_key_usage_aggregates_per_key() -> None:
    db = SessionLocal()
    try:
        user = _make_user(db, email="keyuser@test.local")
        key_a = ApiKey(user_id=user.id, key_prefix="ak_a", key_hash="h-a", name="A")
        key_b = ApiKey(user_id=user.id, key_prefix="ak_b", key_hash="h-b", name="B")
        db.add_all([key_a, key_b])
        db.commit()
        db.refresh(key_a)
        db.refresh(key_b)
        record_token_usage(db, user_id=user.id, api_key_id=key_a.id, input_tokens=10, output_tokens=5)
        record_token_usage(db, user_id=user.id, api_key_id=key_a.id, input_tokens=20, output_tokens=6)
        record_token_usage(db, user_id=user.id, api_key_id=key_b.id, input_tokens=100, output_tokens=50)
        record_token_usage(db, user_id=user.id, input_tokens=999, output_tokens=1)
        usage = get_api_key_usage(db, api_key_ids=[key_a.id, key_b.id])
        assert usage[key_a.id]["record_count"] == 2
        assert usage[key_a.id]["total_input_tokens"] == 30
        assert usage[key_a.id]["total_tokens"] == 30 + 11
        assert usage[key_b.id]["record_count"] == 1
        assert usage[key_b.id]["total_tokens"] == 150
        assert key_a.id in usage and key_b.id in usage
        assert len(usage) == 2
    finally:
        db.close()


def test_api_key_usage_empty_when_no_keys() -> None:
    db = SessionLocal()
    try:
        usage = get_api_key_usage(db, api_key_ids=[99999])
        assert usage == {}
    finally:
        db.close()


def test_user_usage_aggregates_per_user() -> None:
    db = SessionLocal()
    try:
        user_a = _make_user(db, email="ua@test.local")
        user_b = _make_user(db, email="ub@test.local")
        record_token_usage(db, user_id=user_a.id, input_tokens=10, output_tokens=5)
        record_token_usage(db, user_id=user_a.id, input_tokens=20, output_tokens=6)
        record_token_usage(db, user_id=user_b.id, input_tokens=100, output_tokens=50)
        usage = get_user_usage(db, user_ids=[user_a.id, user_b.id])
        assert usage[user_a.id]["record_count"] == 2
        assert usage[user_a.id]["total_input_tokens"] == 30
        assert usage[user_a.id]["total_tokens"] == 41
        assert usage[user_b.id]["record_count"] == 1
        assert usage[user_b.id]["total_tokens"] == 150
        assert len(usage) == 2
    finally:
        db.close()


def test_cleanup() -> None:
    engine.dispose()
    if os.path.exists(_db_path):
        os.unlink(_db_path)
