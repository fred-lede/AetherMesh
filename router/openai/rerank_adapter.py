from __future__ import annotations

from typing import Any

from fastapi import Body

from providers.rerank_adapter import RerankAdapter


def create_rerank_route(service: Any):
    def rerank(payload: dict[str, Any] = Body(...)):
        return service.handle_rerank(payload)
    return rerank


def create_rerank_pool_route():
    def rerank_pool() -> dict[str, Any]:
        return RerankAdapter().pool_summary()
    return rerank_pool
