from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import settings
from providers.base import ProviderAdapter, ProviderError
from providers.http_client import get_session

logger = logging.getLogger("providers.rerank")

_POOL_STATUS: dict[str, dict[str, float]] = {}
COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 120.0
CONNECT_TIMEOUT_S = 3.0

_REGISTRY_CACHE: dict[str, Any] = {"endpoints": [], "fetched_at": 0.0}
REGISTRY_TTL_S = 30.0


def _registry_endpoints() -> list[str]:
    """Rerank endpoints discovered from the cluster registry, local first."""
    now = time.monotonic()
    if now - float(_REGISTRY_CACHE.get("fetched_at", 0.0)) < REGISTRY_TTL_S:
        return list(_REGISTRY_CACHE.get("endpoints", []))

    endpoints: list[str] = []
    try:
        response = requests.get(f"{settings.control_plane_url}/cluster/workers", timeout=3)
        if response.ok:
            payload = response.json()
            workers = payload.get("workers", []) if isinstance(payload, dict) else []
            local_node = settings.local_node_id()
            rerank_workers = [
                worker
                for worker in workers
                if isinstance(worker, dict)
                and (worker.get("provider") == "llama-server" or (worker.get("metadata") or {}).get("kind") == "rerank")
                and worker.get("status") != "dead"
            ]
            rerank_workers.sort(
                key=lambda worker: (str(worker.get("node_id")) != local_node, -int(worker.get("gpu_memory", 0) or 0))
            )
            for worker in rerank_workers:
                try:
                    port = int(worker.get("port", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if not port:
                    continue
                if str(worker.get("node_id")) == local_node:
                    url = f"http://127.0.0.1:{port}"
                else:
                    url = str(worker.get("base_url") or "").strip()
                if url and url not in endpoints:
                    endpoints.append(url.rstrip("/"))
    except (requests.RequestException, ValueError):
        logger.debug("rerank registry lookup failed, falling back to configured pool")

    _REGISTRY_CACHE["endpoints"] = endpoints
    _REGISTRY_CACHE["fetched_at"] = now
    return list(endpoints)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RerankAdapter(ProviderAdapter):
    """Calls a standalone llama.cpp ``--rerank`` server.

    The standard Ollama binary does not expose ``/api/rerank`` on the
    build AetherMesh ships, so rerank models are served by a dedicated
    ``llama-server --rerank`` process. This adapter talks to that
    process's native ``/rerank`` endpoint and converts its llama.cpp
    response into the OpenAI-compatible ``/v1/rerank`` shape.
    """

    provider_name: str = "rerank"

    def __init__(
        self,
        base_url: str | None = None,
        worker: dict[str, Any] | None = None,
        endpoints: list[str] | None = None,
    ) -> None:
        if endpoints:
            pool = [str(url).strip().rstrip("/") for url in endpoints if str(url).strip()]
        elif base_url:
            pool = [base_url.rstrip("/")]
        else:
            pool = settings.rerank_pool_from_env() or _registry_endpoints() or settings.rerank_pool()
        if not pool:
            pool = [settings.rerank_default_base_url.rstrip("/")]
        self.endpoints = pool
        self.base_url = pool[0]
        self.worker = worker

    def _candidate_endpoints(self) -> list[str]:
        """Endpoints to try, in priority order, skipping ones still in cooldown."""
        now = time.monotonic()
        live = [
            url
            for url in self.endpoints
            if _POOL_STATUS.get(url, {}).get("dead_until", 0.0) <= now
        ]
        return live or list(self.endpoints)

    def _mark_dead(self, endpoint: str) -> None:
        state = _POOL_STATUS.get(endpoint, {})
        failures = int(state.get("failures", 0)) + 1
        cooldown = min(COOLDOWN_S * (2 ** (failures - 1)), MAX_COOLDOWN_S)
        _POOL_STATUS[endpoint] = {"failures": failures, "dead_until": time.monotonic() + cooldown}
        logger.warning("rerank endpoint %s marked down for %.0fs (failures=%d)", endpoint, cooldown, failures)

    def _mark_healthy(self, endpoint: str) -> None:
        _POOL_STATUS.pop(endpoint, None)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ProviderError(
            "Rerank provider does not support chat.",
            status_code=400,
            code="unsupported_capability",
        )

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ProviderError(
            "Rerank provider does not support responses.",
            status_code=400,
            code="unsupported_capability",
        )

    def stream(self, payload: dict[str, Any]) -> Any:
        raise ProviderError(
            "Rerank provider does not support streaming.",
            status_code=400,
            code="unsupported_capability",
        )

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ProviderError(
            "Rerank provider does not support embeddings.",
            status_code=400,
            code="unsupported_capability",
        )

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        documents = payload.get("documents", [])
        if not isinstance(documents, list) or not documents:
            raise ProviderError(
                "The request must include a non-empty 'documents' list.",
                status_code=400,
                code="invalid_request",
            )
        if "top_n" in payload:
            try:
                top_n = int(payload.get("top_n"))
            except (TypeError, ValueError):
                top_n = len(documents)
            top_n = max(1, min(top_n, len(documents)))
        else:
            top_n = len(documents)
        body: dict[str, Any] = {
            "query": str(payload.get("query", "")),
            "documents": documents,
            "top_n": top_n,
        }

        last_error: ProviderError | None = None
        for endpoint in self._candidate_endpoints():
            try:
                response = get_session().post(
                    f"{endpoint}/rerank",
                    json=body,
                    timeout=(CONNECT_TIMEOUT_S, settings.request_timeout_s),
                )
            except requests.RequestException as exc:
                self._mark_dead(endpoint)
                last_error = ProviderError(str(exc), status_code=502, code="provider_unreachable")
                logger.warning("rerank endpoint %s unreachable: %s", endpoint, exc)
                continue

            response.encoding = "utf-8"
            if not response.ok:
                self._mark_dead(endpoint)
                last_error = ProviderError(response.text, status_code=response.status_code, code="provider_error")
                logger.warning("rerank endpoint %s returned status %s", endpoint, response.status_code)
                continue

            self._mark_healthy(endpoint)
            data = response.json()
            rows: list[dict[str, Any]] = []
            for item in data.get("results", []):
                if not isinstance(item, dict):
                    continue
                index = _safe_int(item.get("index"), len(rows))
                doc_value = documents[index] if 0 <= index < len(documents) else None
                rows.append(
                    {
                        "index": index,
                        "relevance_score": _safe_float(item.get("relevance_score", item.get("score", 0.0))),
                        "document": doc_value,
                    }
                )

            rows.sort(key=lambda r: r["relevance_score"], reverse=True)
            rows = rows[:top_n]
            return {
                "object": "list",
                "data": rows,
                "model": payload.get("model", self.provider_name),
                "usage": data.get("usage", {}),
                "rerank_via": endpoint,
            }

        if last_error is None:
            raise ProviderError(
                "No rerank endpoint is configured.",
                status_code=503,
                code="worker_unavailable",
            )
        raise last_error

    def health_check(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        active: str | None = None
        for endpoint in self.endpoints:
            try:
                response = get_session().get(f"{endpoint}/health", timeout=5)
            except Exception as exc:  # noqa: BLE001
                entries.append({"base_url": endpoint, "ok": False, "error": str(exc)})
                continue
            if response.ok:
                self._mark_healthy(endpoint)
                entries.append({"base_url": endpoint, "ok": True, "status_code": response.status_code})
                if active is None:
                    active = endpoint
            else:
                entries.append({"base_url": endpoint, "ok": False, "status_code": response.status_code})

        return {
            "ok": active is not None,
            "status_code": 200 if active else 503,
            "base_url": active or self.base_url,
            "endpoints": entries,
        }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default