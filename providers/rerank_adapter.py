from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from providers.base import ProviderAdapter, ProviderError
from providers.http_client import get_session

logger = logging.getLogger("providers.rerank")


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

    def __init__(self, base_url: str | None = None, worker: dict[str, Any] | None = None) -> None:
        self.base_url = (base_url or settings.rerank_default_base_url).rstrip("/")
        self.worker = worker

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

        response = get_session().post(
            f"{self.base_url}/rerank",
            json=body,
            timeout=settings.request_timeout_s,
        )
        response.encoding = "utf-8"
        if not response.ok:
            raise ProviderError(response.text, status_code=response.status_code, code="provider_error")

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
        }

    def health_check(self) -> dict[str, Any]:
        try:
            response = get_session().get(f"{self.base_url}/health", timeout=5)
            return {"ok": response.ok, "status_code": response.status_code, "base_url": self.base_url}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "base_url": self.base_url}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default