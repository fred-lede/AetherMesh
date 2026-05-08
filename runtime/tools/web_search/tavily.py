from __future__ import annotations

import logging
import os
from typing import Any

from runtime.tools.web_search.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
)

logger = logging.getLogger("web_search.tavily")


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            raise SearchProviderError("TAVILY_API_KEY not configured", provider="tavily")

        import requests

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": self._api_key, "query": query, "max_results": max_results},
                timeout=15,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            results: list[SearchResult] = []
            for i, item in enumerate(data.get("results", [])):
                results.append(
                    SearchResult(
                        title=str(item.get("title", "")),
                        url=str(item.get("url", "")),
                        snippet=str(item.get("content", item.get("snippet", ""))),
                        position=i + 1,
                        content=str(item.get("content", "")),
                    )
                )
            return results
        except requests.RequestException as e:
            raise SearchProviderError(
                f"Tavily search failed: {e}", provider="tavily", status_code=getattr(e.response, "status_code", 0)
            ) from e

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        import requests

        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "AetherMesh/1.0"})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            raise SearchProviderError(
                f"Tavily fetch failed: {e}", provider="tavily"
            ) from e
