from __future__ import annotations

import json
from typing import Any

import requests


class AnthropicClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8002", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream, **kwargs}
        resp = self.session.post(f"{self.base_url}/v1/messages", json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data == "[DONE]":
                            break
                        yield json.loads(data)
            return None
        return resp.json()

    def models(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
