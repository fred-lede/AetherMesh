from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings
from providers.registry import CAPABILITY_ALIASES, CANONICAL_CAPABILITIES

logger = logging.getLogger("runtime.orchestration.model_registry_store")

_TOP_LEVEL_KEY = "models"
_cache: list[dict[str, Any]] | None = None
_cache_mtime: float = -1.0


def models_path() -> Path:
    return settings.config_path("models.yaml")


def models_mtime() -> float:
    try:
        return models_path().stat().st_mtime
    except OSError:
        return -1.0


def load_models() -> list[dict[str, Any]]:
    path = models_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        logger.warning("models.yaml is not a mapping; ignoring")
        return []
    models = doc.get(_TOP_LEVEL_KEY) or []
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict)]


def get_models() -> list[dict[str, Any]]:
    global _cache, _cache_mtime
    current = models_mtime()
    if _cache is not None and current == _cache_mtime:
        return _cache
    _cache = load_models()
    _cache_mtime = current
    return _cache


def save_models(models: list[dict[str, Any]]) -> None:
    path = models_path()
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            existing = loaded
    existing[_TOP_LEVEL_KEY] = list(models)
    text = yaml.safe_dump(existing, allow_unicode=True, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    tmp.write_text(text, encoding="utf-8")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            if path.exists():
                os.replace(path, bak)
            os.replace(tmp, path)
            reload_models()
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def reload_models() -> None:
    global _cache, _cache_mtime
    _cache = None
    _cache_mtime = -1.0
    get_models()


LOCAL_PROVIDERS: set[str] = {"ollama", "xtts"}
CLOUD_PROVIDERS: set[str] = {"openai", "gemini", "nvidia_nim", "ollama_cloud"}
_NAME_RE = re.compile(r"^[A-Za-z0-9._:/+-]+$")


def is_local_model(entry: dict[str, Any]) -> bool:
    return str(entry.get("provider", "")).strip() not in CLOUD_PROVIDERS


def _canonical_capability(value: str) -> str:
    key = str(value).strip().lower()
    parsed = CAPABILITY_ALIASES.get(key)
    if parsed is not None:
        return parsed.value
    return key


def normalize_model(entry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": str(entry.get("name", "")).strip(),
        "provider": str(entry.get("provider", "")).strip(),
    }
    if entry.get("worker_bindings") is not None:
        out["worker_bindings"] = [
            {"node_id": str(b.get("node_id", "")).strip(), "port": int(b.get("port", 0))}
            for b in (entry.get("worker_bindings") or [])
        ]
    else:
        out["worker_ports"] = list(entry.get("worker_ports") or [])
    caps = sorted({_canonical_capability(c) for c in (entry.get("capabilities") or [])})
    out["capabilities"] = caps
    vram = entry.get("estimated_vram_mb")
    out["estimated_vram_mb"] = int(vram) if vram is not None else None
    ctx = entry.get("context_length")
    out["context_length"] = int(ctx) if ctx is not None else None
    return out


def validate_model(
    entry: dict[str, Any],
    existing_names: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    name = str(entry.get("name", "")).strip()
    if not name:
        errors.append("name is required")
    elif not _NAME_RE.match(name):
        errors.append(f"invalid name: {name!r}")
    elif existing_names and name in existing_names:
        errors.append(f"duplicate name: {name!r}")

    provider = str(entry.get("provider", "")).strip()
    if not provider:
        errors.append("provider is required")

    bindings = entry.get("worker_bindings")
    if is_local_model(entry):
        if not bindings:
            errors.append("worker_bindings is required for local models")
        else:
            for binding in bindings:
                port = binding.get("port")
                try:
                    port_int = int(port)
                except (TypeError, ValueError):
                    errors.append(f"invalid port: {port!r}")
                    continue
                if not 1 <= port_int <= 65535:
                    errors.append(f"port out of range: {port_int}")

    caps = entry.get("capabilities") or []
    for cap in caps:
        if _canonical_capability(cap) not in CANONICAL_CAPABILITIES:
            errors.append(f"unknown capability: {cap!r}")

    ctx = entry.get("context_length")
    if ctx is not None:
        try:
            if int(ctx) <= 0:
                errors.append("context_length must be a positive integer")
        except (TypeError, ValueError):
            errors.append(f"invalid context_length: {ctx!r}")

    if errors:
        return normalize_model(entry), errors
    return normalize_model(entry), []
