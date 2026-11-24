from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from runtime.orchestration.provider_router import reload_custom_providers

router = APIRouter()


@router.post("/api/providers/reload")
def reload_providers() -> dict[str, Any]:
    reloaded = reload_custom_providers()
    return {"ok": True, "providers": list(reloaded.keys())}
