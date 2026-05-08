from __future__ import annotations

import logging
import re
from typing import Any

from runtime.tools.web_search.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
)

logger = logging.getLogger("web_search.duckduckgo")


class DuckDuckGoSearchProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def configured(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import requests

        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AetherMesh/1.0)",
                    "Accept": "text/html",
                },
            )
            resp.raise_for_status()
            return self._parse_html_results(resp.text, max_results)
        except requests.RequestException as e:
            raise SearchProviderError(
                f"DuckDuckGo search failed: {e}", provider="duckduckgo"
            ) from e

    def _parse_html_results(self, html: str, max_results: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for match in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        ):
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet_match = re.search(
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                html[match.end(): match.end() + 500],
                re.DOTALL,
            )
            snippet = ""
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    position=len(results) + 1,
                )
            )
            if len(results) >= max_results:
                break
        return results

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        import requests

        try:
            resp = requests.get(
                url,
                timeout=timeout_s,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AetherMesh/1.0)",
                },
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            raise SearchProviderError(
                f"DuckDuckGo fetch failed: {e}", provider="duckduckgo"
            ) from e
