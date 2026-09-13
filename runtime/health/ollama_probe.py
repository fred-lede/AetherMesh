from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger("health.ollama_probe")

_EMBED_KEYWORDS = frozenset({"embed", "embedding"})


def _pick_probe_model(loaded: list[dict]) -> str:
    """Pick the smallest non-embedding model from currently loaded models, or
    fall back to the first entry. Embedding-only models cannot run /api/generate
    and would always produce 400 errors for a chat probe."""
    candidates = [
        m for m in loaded
        if not any(kw in m.get("name", "").lower() for kw in _EMBED_KEYWORDS)
    ]
    if not candidates:
        candidates = loaded
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda m: m.get("size_vram", 0),
    ).get("name", "")


@dataclass(slots=True)
class ProbeResult:
    status: str
    detail: str = ""
    latency_ms: int = 0
    model: str = ""
    stale_vram: bool = False


def probe_ollama(
    base_url: str,
    timeout_s: float = 30.0,
    model: str = "",
    session: requests.Session | None = None,
) -> ProbeResult:
    http = session or requests
    base = base_url.rstrip("/")
    started = time.time()

    try:
        resp = http.get(f"{base}/api/tags", timeout=min(timeout_s, 10))
        resp.raise_for_status()
    except requests.RequestException as exc:
        return ProbeResult(status="unreachable", detail=f"/api/tags 失敗：{exc}")

    try:
        ps = http.get(f"{base}/api/ps", timeout=min(timeout_s, 10)).json()
        loaded = ps.get("models") or []
    except (requests.RequestException, ValueError):
        loaded = []

    if not loaded:
        return ProbeResult(status="idle", latency_ms=int((time.time() - started) * 1000))

    stale_vram = all(not m.get("size_vram") for m in loaded)
    if model:
        target = model
    else:
        target = _pick_probe_model(loaded)

    try:
        gen = http.post(
            f"{base}/api/generate",
            json={"model": target, "prompt": "ping", "stream": False, "options": {"num_predict": 1}},
            timeout=timeout_s,
        )
        gen.raise_for_status()
        data = gen.json()
    except requests.RequestException as exc:
        return ProbeResult(
            status="infer_failed",
            detail=f"{target} 推論失敗：{exc}",
            model=target,
            stale_vram=stale_vram,
        )
    except ValueError as exc:
        return ProbeResult(
            status="infer_failed",
            detail=f"{target} 回應非 JSON：{exc}",
            model=target,
            stale_vram=stale_vram,
        )

    if not data.get("done"):
        return ProbeResult(
            status="infer_failed",
            detail=f"{target} 推論未完成（done != true）",
            model=target,
            stale_vram=stale_vram,
        )

    return ProbeResult(
        status="ok",
        latency_ms=int((time.time() - started) * 1000),
        model=target,
        stale_vram=stale_vram,
    )
