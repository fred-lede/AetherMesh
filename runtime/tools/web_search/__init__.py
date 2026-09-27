from __future__ import annotations

import logging
from typing import Any

from config.settings import settings

from runtime.tools.web_search.duckduckgo import DuckDuckGoSearchProvider
from runtime.tools.web_search.exa import ExaSearchProvider
from runtime.tools.web_search.serper import SerperSearchProvider
from runtime.tools.web_search.tavily import TavilySearchProvider
from runtime.tools.web_search.search_provider import SearchProvider, SearchProviderError, SearchResult

logger = logging.getLogger("web_search.manager")

DEFAULT_ORDER = ["exa", "tavily", "serper", "duckduckgo"]


def _order_mtime() -> float:
    try:
        return settings.config_path("web_search.json").stat().st_mtime
    except OSError:
        return -1.0


def _load_order() -> list[str]:
    try:
        data = settings.load_web_search_config()
    except Exception:
        return []
    order = data.get("provider_order")
    return [str(name) for name in order] if isinstance(order, list) else []


class WebSearchManager:
    def __init__(self) -> None:
        self._providers_by_name: dict[str, SearchProvider] = {
            "exa": ExaSearchProvider(),
            "tavily": TavilySearchProvider(),
            "serper": SerperSearchProvider(),
            "duckduckgo": DuckDuckGoSearchProvider(),
        }
        self._order: list[str] = list(DEFAULT_ORDER)
        self._order_mtime: float = -2.0
        self._refresh_order()

    def _refresh_order(self) -> None:
        mtime = _order_mtime()
        if mtime == self._order_mtime:
            return
        requested = _load_order()
        cleaned: list[str] = []
        seen: set[str] = set()
        for name in requested:
            if name in self._providers_by_name and name not in seen:
                cleaned.append(name)
                seen.add(name)
        for name in DEFAULT_ORDER:
            if name not in seen:
                cleaned.append(name)
                seen.add(name)
        self._order = cleaned
        self._order_mtime = mtime

    @property
    def providers(self) -> list[SearchProvider]:
        self._refresh_order()
        return [self._providers_by_name[name] for name in self._order]

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for provider in self.providers:
            if not provider.configured:
                continue
            try:
                logger.debug("Searching via %s: %s", provider.name, query)
                return provider.search(query, max_results=max_results)
            except SearchProviderError as e:
                logger.warning("Search via %s failed: %s", provider.name, e)
                continue
        return self._providers_by_name["duckduckgo"].search(query, max_results=max_results)

    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        for provider in self.providers:
            if not provider.configured:
                continue
            try:
                return provider.fetch_url(url, timeout_s=timeout_s)
            except SearchProviderError:
                continue
        return self._providers_by_name["duckduckgo"].fetch_url(url, timeout_s=timeout_s)


web_search_manager = WebSearchManager()
