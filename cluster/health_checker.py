from __future__ import annotations

from typing import Any

import requests


def http_health(url: str, timeout: int = 3) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "body": response.text[:256],
        }
    except requests.RequestException as exc:
        return {"ok": False, "status_code": 0, "body": str(exc)}


def check_ollama_worker(base_url: str, timeout: int = 3) -> dict[str, Any]:
    return http_health(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
