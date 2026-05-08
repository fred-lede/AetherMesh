from __future__ import annotations

import logging
import os
from typing import Any

from runtime.tools.web_search.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
)

logger = logging.getLogger("web_search.serper")


class SerperSearchProvider(SearchProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("SERPER_API_KEY", "")

    @property
    def name(self) -> str:
        return "serper"

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            raise SearchProviderError("SERPER_API_KEY not configured", provider="serper")

        import requests

        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            results: list[SearchResult] = []

            organic = data.get("organic", [])
            if isinstance(organic, list):
                for i, item in enumerate(organic[:max_results]):
                    results.append(
                        SearchResult(
                            title=str(item.get("title", "")),
                            url=str(item.get("link", "")),
                            snippet=str(item.get("snippet", "")),
                            position=i + 1,
                        )
                    )
            return results
        except requests.RequestException as e:
            raise SearchProviderError(
                f"Serper search failed: {e}", provider="serper", status_code=getattr(e.response, "status_code", 0)
            ) from e

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        import requests

        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "AetherMesh/1.0"})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            raise SearchProviderError(
                f"Serper fetch failed: {e}", provider="serper"
            ) from e
