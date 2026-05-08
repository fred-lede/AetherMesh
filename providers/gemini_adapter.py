from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Iterable

import requests

from config.settings import settings

from .base import ProviderAdapter, ProviderError
from .http_client import get_session, post_with_retry


class GeminiAdapter(ProviderAdapter):
    provider_name = "gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.base_url = os.getenv(
            "GEMINI_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not configured.")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload["model"]
        body = {
            "contents": self._messages_to_contents(payload.get("messages", [])),
            "tools": self._tools_to_gemini(payload.get("tools", [])),
        }
        response = post_with_retry(
            get_session(),
            f"{self.base_url}/models/{model}:generateContent",
            params={"key": self.api_key},
            json=body,
            timeout=settings.request_timeout_s,
        )
        data = response.json()
        text = self._extract_text(data)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        completion = self.chat(payload)
        message = completion["choices"][0]["message"]
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []

        output: list[dict[str, Any]] = []
        if content:
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:16]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            })
        for tc in tool_calls if isinstance(tool_calls, list) else []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            if fn:
                output.append({
                    "type": "tool_call",
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "tool_call_id": str(tc.get("id", "")),
                    "tool_name": str(fn.get("name", "")),
                    "arguments": json.dumps(fn.get("arguments", {}), ensure_ascii=False, separators=(",", ":")),
                    "status": "completed",
                })

        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created": int(time.time()),
            "model": completion["model"],
            "status": "completed",
            "output": output,
            "usage": completion.get("usage", {}),
        }

    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        completion = self.chat(payload)
        message = completion["choices"][0]["message"]
        yield {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": message.get("content", "")},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield "[DONE]"

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload["model"]
        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        rows = []
        for index, value in enumerate(inputs):
            response = post_with_retry(
                get_session(),
                f"{self.base_url}/models/{model}:embedContent",
                params={"key": self.api_key},
                json={"content": {"parts": [{"text": value}]}},
                timeout=settings.request_timeout_s,
            )
            data = response.json()
            rows.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": data.get("embedding", {}).get("values", []),
                }
            )
        return {"object": "list", "data": rows, "model": model, "usage": {}}
    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model", "embedding-001")
        documents = payload.get("documents", [])
        query = str(payload.get("query", ""))
        top_n = payload.get("top_n", len(documents) if documents else 0)

        response = post_with_retry(
            get_session(),
            f"{self.base_url}/models/{model}:rankDocuments",
            params={"key": self.api_key},
            json={
                "model": f"models/{model}",
                "query": {"text": query},
                "documents": [{"text": d} if isinstance(d, str) else d for d in documents],
                "top_n": max(1, int(top_n)),
            },
            timeout=settings.request_timeout_s,
        )
        data = response.json()
        ranks = data.get("ranks", [])
        rows = []
        for r in ranks:
            idx = int(r.get("index", 0))
            doc = documents[idx] if 0 <= idx < len(documents) else None
            rows.append({
                "index": idx,
                "relevance_score": float(r.get("relevance_score", 0.0)),
                "document": doc,
            })
        return {
            "object": "list",
            "data": rows,
            "model": model,
            "usage": {},
        }

    def health_check(self) -> dict[str, Any]:
        response = get_session().get(
            f"{self.base_url}/models",
            params={"key": self.api_key},
            timeout=5,
        )
        response.encoding = "utf-8"
        return {"ok": response.ok, "status_code": response.status_code, "provider": self.provider_name}

    def _messages_to_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents = []
        for message in messages:
            role = "model" if message.get("role") == "assistant" else "user"
            parts = []
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
            else:
                parts.append({"text": str(content)})
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        return contents

    def _tools_to_gemini(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for tool in tools:
            function = tool.get("function", {})
            if not function:
                continue
            declarations.append(
                {
                    "functionDeclarations": [
                        {
                            "name": function.get("name", "tool"),
                            "description": function.get("description", ""),
                            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                        }
                    ]
                }
            )
        return declarations

    def _extract_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

