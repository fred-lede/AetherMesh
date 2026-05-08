from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("web_search")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    @abstractmethod
    def fetch_url(self, url: str, timeout_s: int = 15) -> str:
        ...


class SearchProviderError(RuntimeError):
    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
