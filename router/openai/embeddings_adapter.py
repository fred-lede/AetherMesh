from __future__ import annotations

from typing import Any

from fastapi import Body


def create_embeddings_route(service: Any):
    def embeddings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.handle_embeddings(payload)
    return embeddings
