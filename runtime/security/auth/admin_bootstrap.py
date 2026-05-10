from __future__ import annotations

import logging
import os

from runtime.security.database import SessionLocal, init_db
from runtime.security.models import User
from runtime.security.auth.password import hash_password

logger = logging.getLogger("security.admin_bootstrap")


def bootstrap_admin() -> None:
    init_db()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if admin is not None:
            return

        email = os.getenv("AIIH_ADMIN_EMAIL", "").strip()
        password = os.getenv("AIIH_ADMIN_PASSWORD", "").strip()
        if not email or not password:
            logger.warning(
                "No admin user found and AIIH_ADMIN_EMAIL / AIIH_ADMIN_PASSWORD not set. "
                "Set them in .env to create the initial admin account."
            )
            return

        admin = User(
            email=email,
            password_hash=hash_password(password),
            display_name="Admin",
            role="admin",
        )
        db.add(admin)
        db.commit()
        logger.info("Admin user created: %s", email)
    finally:
        db.close()
