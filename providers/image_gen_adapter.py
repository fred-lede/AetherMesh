from __future__ import annotations

import logging
from typing import Any

from providers.http_client import get_session

logger = logging.getLogger("image_gen_adapter")


class ImageGenAdapter:
    def __init__(self) -> None:
        self.base_url = ""

    def set_worker(self, base_url: str) -> None:
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        n: int = 1,
    ) -> list[str]:
        session = get_session()
        results: list[str] = []
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        for _ in range(n):
            resp = session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            image_b64 = data.get("image", "")
            if image_b64:
                results.append(image_b64)
        return results
