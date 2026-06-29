from __future__ import annotations

import json
import os
from typing import Any, Iterable

import requests

from config.settings import settings

from .base import ProviderAdapter, ProviderError
from .http_client import get_session, post_with_retry


class OpenAIAdapter(ProviderAdapter):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not configured.")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/chat/completions", payload)

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/responses", payload)

    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        body = dict(payload)
        body["stream"] = True
        response = post_with_retry(
            get_session(),
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=settings.request_timeout_s,
            stream=True,
        )
        self._stream_response = response
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                yield "[DONE]"
                return
            try:
                yield json.loads(value)
            except json.JSONDecodeError:
                yield value

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/embeddings", payload)
    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ProviderError("Rerank is not implemented for OpenAI adapter in AIIH yet.")

    def health_check(self) -> dict[str, Any]:
        response = get_session().get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
        return {"ok": response.ok, "status_code": response.status_code, "provider": self.provider_name}

    def abort_stream(self) -> None:
        resp = getattr(self, "_stream_response", None)
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
            self._stream_response = None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = post_with_retry(
            get_session(),
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=payload,
            timeout=settings.request_timeout_s,
        )
        return response.json()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

