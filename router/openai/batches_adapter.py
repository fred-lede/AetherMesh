from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from runtime.orchestration.batch_manager import ALLOWED_BATCH_ENDPOINTS, batch_manager


def create_batches_router(service: Any):
    router = APIRouter(prefix="/v1")

    def _handler_for(endpoint: str):
        if endpoint == "/v1/chat/completions":
            return service.handle_chat
        if endpoint == "/v1/responses":
            return service.handle_responses
        if endpoint == "/v1/embeddings":
            return service.handle_embeddings
        return None

    @router.post("/batches")
    def create_batch(payload: dict[str, Any] = Body(...)):
        endpoint = payload.get("endpoint", "/v1/chat/completions")
        if endpoint not in ALLOWED_BATCH_ENDPOINTS:
            raise HTTPException(status_code=400, detail=f"Unsupported batch endpoint '{endpoint}'")
        input_file_id = payload.get("input_file_id")
        if not input_file_id:
            raise HTTPException(status_code=400, detail="Missing required field 'input_file_id'")
        handler = _handler_for(endpoint)
        if handler is None:
            raise HTTPException(status_code=400, detail=f"Unsupported batch endpoint '{endpoint}'")
        try:
            batch = batch_manager.create_batch(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=payload.get("completion_window", "24h"),
                handler=handler,
                user_id=getattr(payload, "user_id", None),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return batch

    @router.get("/batches/{batch_id}")
    def get_batch(batch_id: str):
        batch = batch_manager._public_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
        return batch

    @router.get("/batches")
    def list_batches(limit: int = 20, after: str | None = None):
        return {
            "object": "list",
            "data": batch_manager.list_batches(limit=limit, after=after),
            "has_more": False,
        }

    @router.post("/batches/{batch_id}/cancel")
    def cancel_batch(batch_id: str):
        batch = batch_manager.cancel_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
        return batch

    return router
