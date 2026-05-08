from __future__ import annotations

from typing import Any

import requests


class OpenAIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8001", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def chat_completions(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        payload = {"model": model, "messages": messages, **kwargs}
        resp = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def embeddings(self, model: str, input_text: str | list[str]) -> dict[str, Any]:
        payload = {"model": model, "input": input_text}
        resp = self.session.post(f"{self.base_url}/v1/embeddings", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def models(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])
