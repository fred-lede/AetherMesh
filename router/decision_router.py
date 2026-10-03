from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from config.settings import settings
from providers.base import ProviderError
from runtime.orchestration import model_registry_store, provider_router

logger = logging.getLogger("router.decision")
router = APIRouter(tags=["decision"])


@router.post("/v1/systemone")
async def systemone(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model is required")
    model = model.strip()

    registry = {"models": model_registry_store.get_models()}
    provider, worker = provider_router.resolve_provider(model, registry)
    if provider != "ollama":
        raise HTTPException(
            status_code=501,
            detail=f"provider '{provider}' does not support decision (/v1/systemone) yet",
        )
    if not (worker or {}).get("base_url"):
        raise HTTPException(status_code=400, detail=f"no Ollama worker is bound for model '{model}'")

    forward = {**payload, "model": settings.strip_model_route_prefix(model)}
    try:
        adapter = provider_router.adapter(provider, worker)
        return await asyncio.to_thread(adapter.systemone, forward)
    except ProviderError as exc:
        code = exc.status_code if 400 <= (exc.status_code or 0) < 500 else 502
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.exception("decision systemone failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
