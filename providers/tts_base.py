from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TTSProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class TTSProviderAdapter(ABC):
    provider_name: str = "tts_base"

    @abstractmethod
    def tts(self, payload: dict[str, Any]) -> bytes:
        ...

    @abstractmethod
    def list_voices(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def register_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def delete_voice(self, voice_id: str) -> bool:
        ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        ...
