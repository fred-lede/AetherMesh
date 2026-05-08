from __future__ import annotations

"""Re-exports from router/openai/responses_adapter for backward compatibility."""

from typing import Any

from fastapi import APIRouter, Body
from router.openai.responses_adapter import create_responses_router

__all__ = ["create_responses_router"]
