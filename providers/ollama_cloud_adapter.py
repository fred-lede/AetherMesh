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


class OllamaCloudAdapter(ProviderAdapter):
    provider_name = "ollama_cloud"

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_CLOUD_API_BASE", "https://ollama.com").rstrip("/")
        self.api_key = os.getenv("OLLAMA_CLOUD_API_KEY", "")
        if not self.api_key:
            raise ProviderError("OLLAMA_CLOUD_API_KEY is not configured.")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._chat_payload(payload, stream=False)
        response = post_with_retry(
            get_session(),
            f"{self.base_url}/api/chat",
            headers=self._headers(),
            json=body,
            timeout=settings.request_timeout_s,
        )
        data = response.json()
        return self._to_chat_completion(payload.get("model", "unknown"), data)

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
        body = self._chat_payload(payload, stream=True)
        response = post_with_retry(
            get_session(),
            f"{self.base_url}/api/chat",
            headers=self._headers(),
            json=body,
            timeout=settings.request_timeout_s,
            stream=True,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = payload.get("model", "unknown")

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            item = json.loads(raw_line)
            if item.get("done"):
                pe = item.get("prompt_eval_count", 0) or 0
                ec = item.get("eval_count", 0) or 0
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": item.get("done_reason", "stop")}],
                    "usage": {
                        "prompt_tokens": pe,
                        "completion_tokens": ec,
                        "total_tokens": pe + ec,
                    },
                }
                yield "[DONE]"
                return

            message = item.get("message", {})
            delta: dict[str, Any] = {"role": "assistant"}

            content = str(message.get("content") or "")
            if content:
                delta["content"] = content

            tool_calls = self._to_openai_tool_calls_for_stream(message.get("tool_calls"))
            if tool_calls:
                delta["tool_calls"] = tool_calls

            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": payload["model"],
            "input": payload.get("input", []),
        }
        response = post_with_retry(
            get_session(),
            f"{self.base_url}/api/embed",
            headers=self._headers(),
            json=body,
            timeout=settings.request_timeout_s,
        )
        data = response.json()
        embeddings = data.get("embeddings", [])
        rows = []
        for index, embedding in enumerate(embeddings):
            rows.append({"object": "embedding", "index": index, "embedding": embedding})
        return {"object": "list", "data": rows, "model": payload["model"], "usage": {}}

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": payload["model"],
            "query": str(payload.get("query", "")),
            "documents": payload.get("documents", []),
        }
        if "top_n" in payload:
            body["top_n"] = payload["top_n"]

        response = post_with_retry(
            get_session(),
            f"{self.base_url}/api/rerank",
            headers=self._headers(),
            json=body,
            timeout=settings.request_timeout_s,
        )

        data = response.json()
        rows: list[dict[str, Any]] = []
        documents = body.get("documents", [])

        for index, item in enumerate(data.get("results", [])):
            if not isinstance(item, dict):
                continue
            doc_index = int(item.get("index", index))
            doc_value = item.get("document")
            if doc_value is None and isinstance(documents, list) and 0 <= doc_index < len(documents):
                doc_value = documents[doc_index]
            rows.append(
                {
                    "index": doc_index,
                    "relevance_score": _safe_float(item.get("relevance_score", item.get("score", 0.0))),
                    "document": doc_value,
                }
            )

        return {
            "object": "list",
            "data": rows,
            "model": payload["model"],
            "usage": data.get("usage", {}),
        }

    def health_check(self) -> dict[str, Any]:
        response = get_session().get(
            f"{self.base_url}/api/tags",
            headers=self._headers(),
            timeout=10,
        )
        response.encoding = "utf-8"
        return {"ok": response.ok, "status_code": response.status_code, "provider": self.provider_name}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_payload(self, payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": payload["model"],
            "messages": self._messages_for_ollama_cloud(payload.get("messages", [])),
            "stream": stream,
        }

        options: dict[str, Any] = dict(payload.get("options", {}))
        max_completion_tokens = payload.get("max_completion_tokens")
        max_tokens = payload.get("max_tokens")
        num_predict = max_completion_tokens if max_completion_tokens is not None else max_tokens
        if num_predict is not None:
            try:
                options["num_predict"] = max(1, int(num_predict))
            except (TypeError, ValueError):
                pass

        for source_key, option_key in (("temperature", "temperature"), ("top_p", "top_p"), ("stop", "stop")):
            if source_key in payload:
                options[option_key] = payload[source_key]

        if options:
            body["options"] = options

        for key in ("tools", "tool_choice", "format", "keep_alive", "think"):
            if key in payload:
                body[key] = payload[key]
        return body

    def _messages_for_ollama_cloud(self, messages: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(self._message_for_ollama_cloud(message))
        return normalized

    def _message_for_ollama_cloud(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {"role": str(message.get("role", "user"))}
        content = message.get("content", "")
        images: list[str] = []

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                text = self._content_part_text_for_ollama_cloud(part)
                if text:
                    text_parts.append(text)
                image = self._content_part_image_for_ollama_cloud(part)
                if image:
                    images.append(image)
            normalized["content"] = "\n".join(text_parts)
        else:
            normalized["content"] = "" if content is None else str(content)

        if images:
            normalized["images"] = images

        tool_calls = message.get("tool_calls")
        if tool_calls:
            normalized["tool_calls"] = tool_calls

        return normalized

    def _content_part_text_for_ollama_cloud(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if not isinstance(part, dict):
            return "" if part is None else str(part)

        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            return str(part.get("text", ""))
        if part_type in {"image_url", "input_image", "image"}:
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
                if url and not url.startswith("data:"):
                    return f"[image: {url}]"
            return ""
        return str(part)

    def _content_part_image_for_ollama_cloud(self, part: Any) -> str | None:
        if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image", "image"}:
            return None
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url", ""))
        else:
            url = str(image_url or part.get("url") or "")
        if not url.startswith("data:"):
            return None
        return url.split(",", 1)[1] if "," in url else url

    def _to_openai_tool_calls_for_stream(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tool_calls):
            if not isinstance(item, dict):
                continue

            function = item.get("function")
            if not isinstance(function, dict):
                continue

            call_id = str(item.get("id") or f"call_{index}")
            name = str(function.get("name") or "")
            arguments = function.get("arguments", "")
            if isinstance(arguments, str):
                arguments_text = arguments
            else:
                arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

            normalized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_text,
                    },
                }
            )

        return normalized

    def _to_chat_completion(self, model: str, data: dict[str, Any]) -> dict[str, Any]:
        message = data.get("message", {})
        tool_calls = self._to_openai_tool_calls(message.get("tool_calls"))
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(message.get("content") or ""),
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": data.get("done_reason", "stop"),
                }
            ],
            "usage": usage,
        }

    def _to_openai_tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tool_calls):
            if not isinstance(item, dict):
                continue

            function = item.get("function")
            if not isinstance(function, dict):
                continue

            name = str(function.get("name") or "")
            arguments = function.get("arguments", "")
            if isinstance(arguments, str):
                arguments_text = arguments
            else:
                arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

            call_id = str(item.get("id") or f"call_{index}")
            normalized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_text,
                    },
                }
            )

        return normalized


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
