from __future__ import annotations

from typing import Any

from fastapi import Body


def create_rerank_route(service: Any):
    def rerank(payload: dict[str, Any] = Body(...)):
        return service.handle_rerank(payload)
    return rerank
