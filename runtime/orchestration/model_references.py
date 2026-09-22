from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings

logger = logging.getLogger("runtime.orchestration.model_references")


def _rules_path() -> Path:
    return settings.config_path("routing_rules.yaml")


def _load_rules() -> dict[str, Any]:
    path = _rules_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def scan_model_references(old_name: str) -> list[dict[str, str]]:
    data = _load_rules()
    found: list[dict[str, str]] = []
    aliases = data.get("aliases") or {}
    if isinstance(aliases, dict):
        for alias, target in aliases.items():
            if str(target) == old_name:
                found.append({"source": "routing_rules", "path": f"aliases.{alias}", "value": old_name})
    fallbacks = data.get("fallbacks") or {}
    if isinstance(fallbacks, dict) and old_name in fallbacks:
        found.append({"source": "routing_rules", "path": f"fallbacks.{old_name}", "value": old_name})
    rules = data.get("rules") or []
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if isinstance(rule, dict) and str(rule.get("model")) == old_name:
                found.append({"source": "routing_rules", "path": f"rules[{index}].model", "value": old_name})
    return found


def _rewrite(node: Any, old_name: str, new_name: str) -> Any:
    if isinstance(node, str):
        return new_name if node == old_name else node
    if isinstance(node, list):
        return [_rewrite(item, old_name, new_name) for item in node]
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            new_key = new_name if key == old_name else key
            result[new_key] = _rewrite(value, old_name, new_name)
        return result
    return node


def update_model_references(old_name: str, new_name: str) -> None:
    path = _rules_path()
    if not path.exists():
        return
    data = _load_rules()
    rewritten = _rewrite(data, old_name, new_name)
    text = yaml.safe_dump(rewritten, allow_unicode=True, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
