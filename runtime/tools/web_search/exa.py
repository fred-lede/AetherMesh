from __future__ import annotations

import logging
import os
from typing import Any

from runtime.tools.web_search.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
)

logger = logging.getLogger("web_search.exa")


class ExaSearchProvider(SearchProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("EXA_API_KEY", "")

    @property
    def name(self) -> str:
        return "exa"

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            raise SearchProviderError("EXA_API_KEY not configured", provider="exa")

        import requests

        try:
            resp = requests.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": max_results,
                    "type": "auto",
                    "contents": {"text": True},
                },
                headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            results: list[SearchResult] = []
            for i, item in enumerate(data.get("results", [])):
                text = str(item.get("text", "") or "")
                title = str(item.get("title", "") or "")
                results.append(
                    SearchResult(
                        title=title,
                        url=str(item.get("url", "") or ""),
                        snippet=text[:500] if text else title,
                        content=text,
                        position=i + 1,
                        metadata={
                            "published_date": item.get("publishedDate"),
                            "author": item.get("author"),
                        },
                    )
                )
            return results
        except requests.RequestException as e:
            raise SearchProviderError(
                f"Exa search failed: {e}",
                provider="exa",
                status_code=getattr(getattr(e, "response", None), "status_code", 0),
            ) from e

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        import requests

        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "AetherMesh/1.0"})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            raise SearchProviderError(f"Exa fetch failed: {e}", provider="exa") from e
