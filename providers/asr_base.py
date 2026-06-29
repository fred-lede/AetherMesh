from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ASRProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class ASRProviderAdapter(ABC):
    provider_name: str = "asr_base"

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        ...
