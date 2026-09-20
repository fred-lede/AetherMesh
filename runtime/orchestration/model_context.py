from __future__ import annotations

import logging
import threading
from typing import Any

import requests

from config.settings import settings
from runtime.intelligence.provider_scoring import provider_capability_registry

logger = logging.getLogger("orchestration.model_context")

_CTX_FIELD_NAMES = (
    "context_window",
    "context_length",
    "max_context_window",
    "max_context_length",
    "max_model_len",
    "max_sequence_length",
    "num_ctx",
)

_lock = threading.Lock()
_auto_ctx: dict[str, int] = {}


def _model_entries() -> list[dict[str, Any]]:
    registry = settings.model_registry()
    if isinstance(registry, dict):
        models = registry.get("models", [])
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
    return []


def configured_context_length(model: str) -> int | None:
    for entry in _model_entries():
        if entry.get("name") == model:
            try:
                val = int(entry.get("context_length", 0))
            except (TypeError, ValueError):
                val = 0
            if val > 0:
                return val
            break
    return None


def resolve_context_length(model: str) -> int | None:
    configured = configured_context_length(model)
    if configured is not None:
        return configured
    with _lock:
        return _auto_ctx.get(model)


def resolve_effective_max_context(provider: str, model: str | None = None) -> int:
    if model:
        model_ctx = resolve_context_length(model)
        if model_ctx:
            return model_ctx
    caps = provider_capability_registry.get_capabilities(provider)
    if caps:
        return caps.max_context
    return 8192


def _extract_context_length(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in _CTX_FIELD_NAMES:
        val = payload.get(key)
        if val is not None:
            try:
                parsed = int(val)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _extract_ollama_model_info_ctx(model_info: dict[str, Any]) -> int | None:
    best: int | None = None
    for key, val in model_info.items():
        if not key.endswith("context_length") or not isinstance(val, (int, str)):
            continue
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            continue
        if parsed <= 0:
            continue
        if key.startswith("general."):
            best = best or parsed
        else:
            return parsed
    return best


def fetch_context_length(
    model: str,
    provider: str = "ollama",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> int | None:
    if provider == "ollama":
        url = f"{base_url.rstrip('/')}/api/show"
        try:
            resp = requests.post(url, json={"model": model}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("fetch ollama ctx for %s failed: %s", model, exc)
            return None
        if not isinstance(data, dict):
            return None
        direct = _extract_context_length(data)
        if direct:
            return direct
        params = data.get("parameters") or {}
        if isinstance(params, dict) and params:
            return _extract_context_length(params)
        model_info = data.get("model_info")
        if isinstance(model_info, dict):
            arch_ctx = _extract_ollama_model_info_ctx(model_info)
            if arch_ctx:
                return arch_ctx
        return None

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("fetch cloud ctx for %s failed: %s", model, exc)
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get("data")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("id", "")).strip() == model:
                return _extract_context_length(entry)
    return _extract_context_length(data)


def refresh_auto_context(models: list[dict[str, Any]]) -> dict[str, int]:
    updated: dict[str, int] = {}
    for entry in models:
        model = str(entry.get("name", "")).strip()
        if not model or configured_context_length(model) is not None:
            continue
        provider = str(entry.get("provider", "ollama"))
        if provider == "ollama":
            base_urls = _ollama_base_urls(entry)
        else:
            single = _base_url_for(entry, provider)
            base_urls = [single] if single else []
        if not base_urls:
            logger.debug("skip ctx fetch for %s: no base_url", model)
            continue
        ctx = None
        for base_url in base_urls:
            ctx = fetch_context_length(model, provider=provider, base_url=base_url)
            if ctx:
                break
        if ctx:
            updated[model] = ctx
            logger.info("auto context for %s: %d", model, ctx)
    if updated:
        with _lock:
            _auto_ctx.update(updated)
    return updated


def _base_url_for(entry: dict[str, Any], provider: str) -> str | None:
    if provider == "ollama":
        urls = _ollama_base_urls(entry)
        return urls[0] if urls else None
    if provider == "gemini":
        return settings.gemini_base_url if hasattr(settings, "gemini_base_url") else "https://generativelanguage.googleapis.com/v1beta"
    if provider == "openai":
        return getattr(settings, "openai_base_url", None) or "https://api.openai.com/v1"
    if provider == "nvidia_nim":
        return "https://integrate.api.nvidia.com/v1"
    if provider == "ollama_cloud":
        return "https://ollama.com/api"
    return None


def _ollama_base_urls(entry: dict[str, Any]) -> list[str]:
    bindings = entry.get("worker_bindings")
    urls: list[str] = []
    if not isinstance(bindings, list):
        return urls
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        try:
            url = settings.worker_base_url(binding)
        except Exception as exc:
            logger.debug("worker_base_url error for %s: %s", binding.get("node_id"), exc)
            url = None
        if url:
            urls.append(url)
    return urls