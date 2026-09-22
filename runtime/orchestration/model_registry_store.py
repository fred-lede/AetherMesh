from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings

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
