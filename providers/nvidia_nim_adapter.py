from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter

from .base import ProviderAdapter, ProviderError


class NvidiaNIMAdapter(ProviderAdapter):
    provider_name = "nvidia_nim"
    _queue_lock = threading.Lock()
    _next_request_at = 0.0

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
        self.timeout_s = int(os.getenv("NVIDIA_NIM_TIMEOUT", "12"))
        self.min_interval_s = float(os.getenv("NVIDIA_NIM_MIN_INTERVAL", "1.5"))
        self.max_retries = int(os.getenv("NVIDIA_NIM_MAX_RETRIES", "2"))
        self.backoff_factor = float(os.getenv("NVIDIA_NIM_BACKOFF_FACTOR", "2.0"))
        self.retryable_statuses = {429, 502, 503, 504}
        self._tool_name_map: dict[str, str] = {}
        self._session = requests.Session()
        self._session.mount("http://", HTTPAdapter(max_retries=0))
        self._session.mount("https://", HTTPAdapter(max_retries=0))
        if not self.api_key:
            raise ProviderError("NVIDIA_NIM_API_KEY is not configured.")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        completion = self._post_json("/chat/completions", self._chat_payload(payload, stream=False))
        self._restore_completion_names(completion)
        return completion

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        stripped = self._strip_prefix(payload)
        if "messages" in stripped:
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
        return self._post_json("/responses", stripped)

    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        body = self._chat_payload(payload, stream=True)
        response = self._post_with_retry("/chat/completions", body, stream=True)
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
                chunk = json.loads(value)
            except json.JSONDecodeError:
                yield value
                continue
            self._restore_chunk_names(chunk)
            yield chunk

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/embeddings", self._strip_prefix(payload))

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/ranking", self._strip_prefix(payload))

    def health_check(self) -> dict[str, Any]:
        self._wait_for_turn()
        response = self._session.get(f"{self.base_url}/models", headers=self._headers(), timeout=15)
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
        response = self._post_with_retry(path, payload, stream=False)
        return response.json()

    def _post_with_retry(
        self, path: str, payload: dict[str, Any], stream: bool = False,
    ) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_turn()
            try:
                response = self._session.post(
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_s,
                    stream=stream,
                )
            except requests.Timeout as exc:
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise self._timeout_error(exc) from exc
            response.encoding = "utf-8"
            if response.ok:
                return response
            if response.status_code in self.retryable_statuses and attempt < self.max_retries:
                delay = self.backoff_factor ** attempt
                time.sleep(delay)
                last_exc = self._provider_error(response)
                continue
            raise self._provider_error(response)
        if isinstance(last_exc, requests.Timeout):
            raise self._timeout_error(last_exc) from last_exc
        if isinstance(last_exc, ProviderError):
            raise last_exc
        raise ProviderError(f"NVIDIA NIM request failed after {self.max_retries + 1} attempts")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _strip_prefix(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        model = body.get("model", "")
        prefix = f"{self.provider_name}/"
        if model.startswith(prefix):
            body["model"] = model[len(prefix):]
        return body

    _CHAT_KEYS = frozenset({
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "stream",
        "stream_options",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
    })

    def _chat_payload(self, payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        body = {key: payload[key] for key in self._CHAT_KEYS if key in payload}
        if stream:
            body["stream"] = True
        tools = body.get("tools")
        name_map: dict[str, str] = {}
        if isinstance(tools, list):
            filtered: list[dict[str, Any]] = []
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                fn = tool.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    original = str(fn_copy.get("name") or "")
                    if original:
                        sanitized = self._sanitize_tool_name(original)
                        if sanitized != original:
                            fn_copy["name"] = sanitized
                            name_map[sanitized] = original
                    filtered.append({"type": "function", "function": fn_copy})
                else:
                    filtered.append(tool)
            body["tools"] = filtered
        self._tool_name_map = name_map
        if "tool_choice" in body and not body.get("tools"):
            body.pop("tool_choice")
        model = str(body.get("model", ""))
        prefix = f"{self.provider_name}/"
        if model.startswith(prefix):
            body["model"] = model[len(prefix):]
        return body

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        out: list[str] = []
        for char in str(name or ""):
            if char == ".":
                out.append("__")
            elif char.isascii() and (char.isalnum() or char in "_-"):
                out.append(char)
            else:
                out.append("_")
        return "".join(out)

    def _restore_tool_name(self, name: str) -> str:
        return self._tool_name_map.get(name, name)

    def _restore_completion_names(self, completion: dict[str, Any]) -> None:
        if not self._tool_name_map:
            return
        for choice in completion.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for tc in message.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict) and str(fn.get("name") or "") in self._tool_name_map:
                        fn["name"] = self._tool_name_map[str(fn["name"])]

    def _restore_chunk_names(self, chunk: dict[str, Any]) -> None:
        if not self._tool_name_map:
            return
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            for tc in delta.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict) and str(fn.get("name") or "") in self._tool_name_map:
                        fn["name"] = self._tool_name_map[str(fn["name"])]

    def _wait_for_turn(self) -> None:
        with self._queue_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            if delay:
                time.sleep(delay)
            self.__class__._next_request_at = time.monotonic() + self.min_interval_s

    def _defer_after_limit(self, retry_after: int | None) -> None:
        delay = float(retry_after or 60)
        with self._queue_lock:
            self.__class__._next_request_at = max(self._next_request_at, time.monotonic() + delay)

    def _timeout_error(self, exc: requests.Timeout) -> ProviderError:
        return ProviderError(
            f"NVIDIA NIM provider_timeout after {self.timeout_s}s: {exc}",
            status_code=504,
            code="provider_timeout",
        )

    def _provider_error(self, response: requests.Response) -> ProviderError:
        status_code = int(response.status_code)
        retry_after = self._retry_after(response)
        message = self._error_message(response)
        code = "provider_error"
        if status_code == 429:
            code = "provider_rate_limited"
            if retry_after is None:
                retry_after = 60
            self._defer_after_limit(retry_after)
        elif status_code == 404:
            code = "model_not_found"
        elif status_code in {502, 503, 504}:
            code = "provider_overloaded"
        elif status_code == 408:
            code = "provider_timeout"

        return ProviderError(
            f"NVIDIA NIM {code}: {message}",
            status_code=status_code,
            retry_after=retry_after,
            code=code,
        )

    def _error_message(self, response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text or response.reason
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            detail = data.get("detail")
            if detail:
                return str(detail)
            message = data.get("message")
            if message:
                return str(message)
        return response.text or response.reason

    def _retry_after(self, response: requests.Response) -> int | None:
        value = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(1, int(float(value)))
        except (TypeError, ValueError):
            return None
