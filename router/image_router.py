from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Response, UploadFile

from config.settings import settings
from runtime.orchestration.provider_router import adapter as get_adapter

logger = logging.getLogger("router.image_gen")
router = APIRouter(tags=["image_gen"])


def _resolve_adapter():
    if not settings.image_gen_enabled:
        raise HTTPException(status_code=503, detail="Image generation is not enabled")
    adapter = get_adapter("image_gen")
    if adapter is None:
        raise HTTPException(status_code=503, detail="Image gen adapter unavailable (cooling down)")
    return adapter


def _generate(
    adapter: Any,
    model: str,
    prompt: str,
    n: int,
) -> list[str]:
    return adapter.generate(model, prompt, n=n)


@router.post("/v1/images/generations")
async def create_image(payload: dict[str, Any], response: Response) -> dict[str, Any]:
    response.headers["Connection"] = "close"
    model = payload.get("model", settings.image_gen_default_model)
    prompt = payload.get("prompt", "")
    n = payload.get("n", 1)

    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")

    adapter = _resolve_adapter()
    try:
        images = await asyncio.to_thread(_generate, adapter, model, prompt, n)
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "created": int(time.time()),
        "data": [{"b64_json": img} for img in images],
    }


@router.post("/v1/images/edits")
async def create_image_edit(
    response: Response,
    image: UploadFile,
    prompt: str = Form(...),
    model: str = Form(None),
    n: int = Form(1),
) -> dict[str, Any]:
    response.headers["Connection"] = "close"
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    resolved_model = model or settings.image_gen_default_model
    if n < 1 or n > 10:
        raise HTTPException(status_code=400, detail="n must be between 1 and 10")

    adapter = _resolve_adapter()
    try:
        images = await asyncio.to_thread(_generate, adapter, resolved_model, prompt, n)
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "created": int(time.time()),
        "data": [{"b64_json": img} for img in images],
    }
