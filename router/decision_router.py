from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from providers.base import ProviderError
from runtime.orchestration import model_registry_store, provider_router

logger = logging.getLogger("router.decision")
router = APIRouter(tags=["decision"])


@router.post("/v1/systemone")
async def systemone(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    registry = {"models": model_registry_store.get_models()}
    provider, worker = provider_router.resolve_provider(model, registry)
    if provider != "ollama":
        raise HTTPException(
            status_code=501,
            detail=f"provider '{provider}' does not support decision (/v1/systemone) yet",
        )
    adapter = provider_router.adapter(provider, worker)
    try:
        return await asyncio.to_thread(adapter.systemone, payload)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("decision systemone failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
