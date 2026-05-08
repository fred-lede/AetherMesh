from __future__ import annotations

import os
from typing import Any


class APIKeyAuth:
    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._load_from_env()

    def _load_from_env(self) -> None:
        key = os.environ.get("AIIH_API_KEY", "")
        if key:
            self._keys.add(key)

    def configure(self, api_keys: list[str]) -> None:
        for key in api_keys:
            if key:
                self._keys.add(key)

    def validate(self, key: str) -> bool:
        if not self._keys:
            return True
        return key in self._keys

    def add_key(self, key: str) -> None:
        if key:
            self._keys.add(key)

    def remove_key(self, key: str) -> None:
        self._keys.discard(key)

    @property
    def enabled(self) -> bool:
        return len(self._keys) > 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "key_count": len(self._keys),
        }


api_key_auth = APIKeyAuth()
