from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: int | None = None,
        code: str = "provider_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.code = code


class ProviderAdapter(ABC):
    provider_name: str = "base"

    @abstractmethod
    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        raise NotImplementedError

    @abstractmethod
    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError
