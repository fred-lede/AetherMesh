from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from runtime.asr.asr_manager import ASRConfigError, LOCAL_ENGINES, asr_manager

router = APIRouter()


def _detail(exc: ASRConfigError) -> dict[str, Any]:
    detail: dict[str, Any] = {"message": exc.message}
    if exc.details is not None:
        detail.update(exc.details)
    return detail


@router.get("/api/asr/config")
def get_asr_config() -> dict[str, Any]:
    snapshot = asr_manager.snapshot()
    snapshot["targets"] = asr_manager.list_targets()
    snapshot["engines"] = {provider: list(models) for provider, models in LOCAL_ENGINES.items()}
    return snapshot


@router.post("/api/asr/config")
def post_asr_config(body: dict[str, Any] = Body(...)) -> JSONResponse:
    target_id = body.get("target_id")
    provider = body.get("provider")
    model = body.get("model")
    if target_id is not None and not isinstance(target_id, str):
        raise HTTPException(status_code=400, detail={"message": "target_id must be a string"})
    if provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=400, detail={"message": "provider must be a string"})
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail={"message": "model must be a string"})
    try:
        snapshot = asr_manager.reload(
            str(target_id or ""),
            provider=provider,
            model=model,
        )
    except ASRConfigError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_detail(exc)) from exc
    return JSONResponse(status_code=202, content=snapshot)
