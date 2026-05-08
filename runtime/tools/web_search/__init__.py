from __future__ import annotations

import logging
from typing import Any

from runtime.tools.web_search.duckduckgo import DuckDuckGoSearchProvider
from runtime.tools.web_search.serper import SerperSearchProvider
from runtime.tools.web_search.tavily import TavilySearchProvider
from runtime.tools.web_search.search_provider import SearchProvider, SearchProviderError, SearchResult

logger = logging.getLogger("web_search.manager")


class WebSearchManager:
    def __init__(self) -> None:
        self._providers: list[SearchProvider] = [
            TavilySearchProvider(),
            SerperSearchProvider(),
            DuckDuckGoSearchProvider(),
        ]

    @property
    def providers(self) -> list[SearchProvider]:
        return list(self._providers)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        errors: list[str] = []
        for provider in self._providers:
            if not provider.configured:
                continue
            try:
                logger.debug("Searching via %s: %s", provider.name, query)
                return provider.search(query, max_results=max_results)
            except SearchProviderError as e:
                logger.warning("Search via %s failed: %s", provider.name, e)
                errors.append(f"{provider.name}: {e}")
                continue

        return DuckDuckGoSearchProvider().search(query, max_results=max_results)

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        for provider in self._providers:
            if not provider.configured:
                continue
            try:
                return provider.fetch_url(url, timeout_s=timeout_s)
            except SearchProviderError:
                continue
        return DuckDuckGoSearchProvider().fetch_url(url, timeout_s=timeout_s)


web_search_manager = WebSearchManager()
