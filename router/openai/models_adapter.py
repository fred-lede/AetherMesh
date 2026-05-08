from __future__ import annotations

from typing import Any


def create_models_route(service: Any):
    def models() -> dict[str, Any]:
        return service.list_models()
    return models
