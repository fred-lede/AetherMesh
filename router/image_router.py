from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Response, UploadFile

from config.settings import settings
from runtime.orchestration import model_registry_store, provider_router
from runtime.orchestration.provider_router import adapter as get_adapter

logger = logging.getLogger("router.image_gen")
router = APIRouter(tags=["image_gen"])

_CLOUD_IMAGE_PROVIDERS = {"openai", "gemini", "nvidia_nim", "ollama_cloud"}


def _resolve_image_provider(model: str) -> str | None:
    try:
        registry = {"models": model_registry_store.get_models()}
        provider, _worker = provider_router.resolve_provider(model, registry)
    except Exception:
        return None
    if provider in _CLOUD_IMAGE_PROVIDERS or provider_router.is_custom_provider(provider):
        return provider
    return None


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

    provider = _resolve_image_provider(model)
    if provider is not None:
        adapter = provider_router.adapter(provider)
        if adapter is None:
            raise HTTPException(status_code=503, detail=f"Provider '{provider}' adapter unavailable")
        if not hasattr(adapter, "images"):
            raise HTTPException(status_code=501, detail=f"Provider '{provider}' does not support image generation yet")
        provider_payload = {**payload, "model": model, "prompt": prompt, "n": n}
        try:
            result = await asyncio.to_thread(adapter.images, provider_payload)
        except Exception as exc:
            logger.exception("Cloud image generation failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))
        if isinstance(result, dict):
            return {"created": result.get("created", int(time.time())), "data": result.get("data") or []}
        return {"created": int(time.time()), "data": []}

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

    if _resolve_image_provider(resolved_model) is not None:
        raise HTTPException(status_code=501, detail="Cloud image edits are not supported yet")

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
