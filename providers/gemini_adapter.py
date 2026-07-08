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

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = (
            base_url or os.getenv(
                "GEMINI_API_BASE",
                "https://generativelanguage.googleapis.com/v1beta",
            )
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
        return self._to_chat_completion(data, model)

    def _to_chat_completion(self, data: dict[str, Any], model: str) -> dict[str, Any]:
        text, tool_calls = self._extract_content(data)
        um = data.get("usageMetadata") or {}
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": um.get("promptTokenCount", 0),
                "completion_tokens": um.get("candidatesTokenCount", 0),
                "total_tokens": um.get("totalTokenCount", 0),
            },
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
        model = payload["model"]
        body = {
            "contents": self._messages_to_contents(payload.get("messages", [])),
            "tools": self._tools_to_gemini(payload.get("tools", [])),
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        response = post_with_retry(
            get_session(),
            f"{self.base_url}/models/{model}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
            json=body,
            timeout=settings.request_timeout_s,
            stream=True,
        )
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        last_usage_metadata: dict[str, Any] = {}
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or raw_line.startswith(":"):
                continue
            if raw_line.startswith("data: "):
                raw_line = raw_line[6:]
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            um = data.get("usageMetadata")
            if um:
                last_usage_metadata = um
            candidates = data.get("candidates", [])
            if not candidates:
                continue
            content_obj = candidates[0].get("content", {})
            parts = content_obj.get("parts", [])
            delta: dict[str, Any] = {"role": "assistant"}
            text_parts: list[str] = []
            tc_list: list[dict[str, Any]] = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                if "functionCall" in part:
                    fc = part["functionCall"]
                    idx = len(tc_list)
                    tc_list.append({
                        "id": f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False, separators=(",", ":")),
                        },
                    })
            if text_parts:
                delta["content"] = "".join(text_parts)
            if tc_list:
                delta["tool_calls"] = tc_list
            if "content" in delta or "tool_calls" in delta:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            if tc_list:
                finish_reason = "tool_calls"
        usage = {
            "prompt_tokens": last_usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": last_usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": last_usage_metadata.get("totalTokenCount", 0),
        } if last_usage_metadata else {}
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": usage,
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
            images_from_field: list[str] = list(message.get("images") or [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        gemini_part = self._image_url_to_gemini_inline(item)
                        if gemini_part:
                            parts.append(gemini_part)
                    elif isinstance(item, dict) and item.get("type") in ("input_audio", "audio"):
                        gemini_part = self._input_audio_to_gemini_inline(item)
                        if gemini_part:
                            parts.append(gemini_part)
            else:
                parts.append({"text": str(content)})
            for img_data in images_from_field:
                parts.append({"inline_data": {"mime_type": "image/png", "data": img_data}})
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        return contents

    @staticmethod
    def _image_url_to_gemini_inline(part: dict[str, Any]) -> dict[str, Any] | None:
        image_url = part.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = str(image_url.get("url", ""))
        if not url:
            return None
        mime_type = "image/png"
        data = url
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            if ";" in header:
                mime_type = header.split(":", 1)[1].split(";", 1)[0]
            if not data:
                return None
        return {"inline_data": {"mime_type": mime_type, "data": data}}

    @staticmethod
    def _input_audio_to_gemini_inline(part: dict[str, Any]) -> dict[str, Any] | None:
        audio = part.get("input_audio") or part
        if not isinstance(audio, dict):
            return None
        data = audio.get("data")
        if not data:
            return None
        fmt = str(audio.get("format", "wav"))
        mime_type = f"audio/{fmt}"
        return {"inline_data": {"mime_type": mime_type, "data": str(data)}}

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

    def _extract_content(self, data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        candidates = data.get("candidates", [])
        if not candidates:
            return "", []
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for idx, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False, separators=(",", ":")),
                    },
                })
        return "".join(text_parts), tool_calls

