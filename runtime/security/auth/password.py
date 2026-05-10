from __future__ import annotations

import hashlib
import os


def hash_password(password: str) -> str:
    salt = os.urandom(32)
    pwd = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=64,
    )
    return salt.hex() + ":" + pwd.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, pwd_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(pwd_hex)
        pwd = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=64,
        )
        return pwd == expected
    except (ValueError, TypeError):
        return False
