from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body


def create_responses_router(service: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    def responses(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.handle_responses(payload)

    @router.get("/v1/models")
    def models() -> dict[str, Any]:
        return service.list_models()

    @router.post("/v1/embeddings")
    def embeddings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.handle_embeddings(payload)

    return router
