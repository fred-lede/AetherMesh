from __future__ import annotations

"""OpenAI-compatible API router. Re-exports from router/openai/ adapters."""

import asyncio
from pathlib import Path
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from config.settings import settings

if settings.debug_responses:
    logging.getLogger().setLevel(logging.INFO)
from runtime.orchestration.openai_handler import RouterService
from runtime.security.middleware import add_security_middleware
from runtime.gpu_os.routes import gpu_router
from runtime.multi_agent.routes import agent_router
from router.openai.chat_adapter import create_chat_completions_route
from router.openai.responses_adapter import create_responses_router
from router.openai.models_adapter import create_models_route
from router.openai.embeddings_adapter import create_embeddings_route
from router.openai.rerank_adapter import create_rerank_route
from router.files_router import router as files_router
from runtime.tools.file_cleanup import ensure_cleanup_dir, get_file_cleanup_manager
from router.skills_router import skills_router
from runtime.skills.skill_registry import skill_registry

logger = logging.getLogger("openai_router")


service = RouterService()
app = FastAPI(title="AetherMesh Router", version="4.0.0")

if settings.debug_responses:
    logging.getLogger("uvicorn.error").warning(
        "openai_router.env AIIH_DEBUG_RESPONSES=true"
    )

add_security_middleware(app, enable_rate_limit=settings.rate_limit_enabled)

_cleanup_task: asyncio.Task | None = None


@app.on_event("startup")
async def _startup_file_cleanup() -> None:
    global _cleanup_task
    ensure_cleanup_dir()
    mgr = get_file_cleanup_manager()
    _cleanup_task = asyncio.create_task(mgr.background_cleanup_loop())


@app.on_event("shutdown")
async def _shutdown_file_cleanup() -> None:
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None


app.include_router(gpu_router)
app.include_router(agent_router)
app.include_router(files_router)
app.include_router(skills_router)
app.include_router(create_responses_router(service))


def _error_type_for_status(status_code: int) -> str:
    if status_code == 429:
        return "rate_limit_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "server_error"


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = int(exc.status_code)
    detail = exc.detail
    headers = dict(exc.headers or {})

    code = "internal_error"
    message = "Request failed."
    extra: dict[str, Any] = {}

    if isinstance(detail, dict):
        code = str(detail.get("code") or code)
        message = str(detail.get("message") or message)
        if "retry_after" in detail:
            extra["retry_after"] = detail["retry_after"]
        if "param" in detail:
            extra["param"] = detail["param"]
    elif isinstance(detail, str):
        message = detail
    elif detail is not None:
        message = str(detail)

    error_payload: dict[str, Any] = {
        "message": message,
        "type": _error_type_for_status(status_code),
        "code": code,
    }
    error_payload.update(extra)

    body: dict[str, Any] = {"error": error_payload}
    if request.url.path.startswith("/cluster/"):
        body = {"detail": {"code": code, "message": message, **extra}}

    return JSONResponse(status_code=status_code, content=body, headers=headers)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "router"}


@app.get("/.well-known/ai-plugin.json")
def ai_plugin_manifest() -> dict[str, Any]:
    skills = skill_registry.list_skills()
    return {
        "schema_version": "v1",
        "name_for_human": "AetherMesh Skills",
        "name_for_model": "aethermesh_skills",
        "description_for_human": "Built-in AetherMesh runtime skills including web search, file operations, code execution, and shell commands.",
        "description_for_model": "Plugin providing AetherMesh runtime skills: web_search, web_fetch, code_interpreter, shell_commands, file_operations, plugin_manager.",
        "auth": {"type": "none"},
        "api": {"type": "openapi", "url": "/openapi.json", "is_user_authenticated": False},
        "contact_email": "support@aethermesh.local",
        "legal_info_url": "http://aethermesh.local/legal",
        "skills": [_descriptor_to_dict(s) for s in skills],
    }


def _descriptor_to_dict(s: Any) -> dict[str, Any]:
    return {
        "name": s.name,
        "description": s.description,
        "capabilities": s.capabilities or [],
        "parameters": s.parameters or {},
    }


@app.post("/v1/chat/completions")
def chat_completions(request: Request, payload: dict[str, Any] = Body(...)):
    return create_chat_completions_route(service)(request, payload)


@app.post("/v1/rerank")
def rerank(payload: dict[str, Any] = Body(...)):
    return create_rerank_route(service)(payload)


@app.post("/v1/embeddings")
def embeddings(payload: dict[str, Any] = Body(...)):
    return create_embeddings_route(service)(payload)


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return create_models_route(service)()
