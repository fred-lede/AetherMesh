from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger("security.database")

_DB_PATH: Path | None = None


def db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        env_path = os.getenv("AIIH_DB_PATH", "").strip()
        if env_path:
            _DB_PATH = Path(env_path)
        else:
            config_dir = Path(__file__).resolve().parent.parent.parent / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            _DB_PATH = config_dir / "aiih.db"
    return _DB_PATH


def db_url() -> str:
    return f"sqlite:///{db_path()}"


engine = create_engine(db_url(), echo=False, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from runtime.security.models import User, ApiKey, Session

    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        import sqlalchemy
        try:
            row = conn.execute(
                sqlalchemy.text("PRAGMA table_info(users)")
            ).fetchall()
            cols = {r[1] for r in row}
            if "must_change_password" not in cols:
                conn.execute(
                    sqlalchemy.text("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN DEFAULT 0")
                )
                conn.commit()
                logger.info("Added must_change_password column to users table")
        except Exception:
            pass

    logger.info("Database initialized at %s", db_path())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_sync() -> Any:
    return SessionLocal()
