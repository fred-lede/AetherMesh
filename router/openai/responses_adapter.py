from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse


def create_responses_router(service: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    async def responses(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        stream = payload.get("stream", False)
        if stream:
            generator = service.handle_streaming_responses(payload)
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return service.handle_responses(payload)

    @router.get("/v1/responses/{response_id}")
    def get_response(response_id: str) -> dict[str, Any]:
        return service.get_response(response_id)

    @router.delete("/v1/responses/{response_id}")
    def delete_response(response_id: str) -> dict[str, Any]:
        return service.delete_response(response_id)

    @router.patch("/v1/responses/{response_id}")
    def update_response(response_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.update_response(response_id, payload)

    return router
