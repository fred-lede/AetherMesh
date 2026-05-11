from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import StreamingResponse


def create_responses_router(service: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    async def responses(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        user_id = getattr(request.state, "user_id", None)
        api_key_id = getattr(request.state, "api_key_id", None)
        stream = payload.get("stream", False)
        if stream:
            generator = service.handle_streaming_responses(payload, user_id=user_id, api_key_id=api_key_id)
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return service.handle_responses(payload, user_id=user_id, api_key_id=api_key_id)

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
