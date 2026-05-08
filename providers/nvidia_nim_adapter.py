from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter

from .base import ProviderAdapter, ProviderError


class NvidiaNIMAdapter(ProviderAdapter):
    provider_name = "nvidia_nim"
    _queue_lock = threading.Lock()
    _next_request_at = 0.0

    def __init__(self) -> None:
        self.base_url = os.getenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.api_key = os.getenv("NVIDIA_NIM_API_KEY", "")
        self.timeout_s = int(os.getenv("NVIDIA_NIM_TIMEOUT", "12"))
        self.min_interval_s = float(os.getenv("NVIDIA_NIM_MIN_INTERVAL", "1.5"))
        self._session = requests.Session()
        self._session.mount("http://", HTTPAdapter(max_retries=0))
        self._session.mount("https://", HTTPAdapter(max_retries=0))
        if not self.api_key:
            raise ProviderError("NVIDIA_NIM_API_KEY is not configured.")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/chat/completions", self._strip_prefix(payload))

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
        body = self._strip_prefix(payload)
        body["stream"] = True
        self._wait_for_turn()
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=self.timeout_s,
                stream=True,
            )
        except requests.Timeout as exc:
            raise self._timeout_error(exc) from exc
        response.encoding = "utf-8"
        if not response.ok:
            raise self._provider_error(response)
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
        return self._post_json("/embeddings", self._strip_prefix(payload))

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/ranking", self._strip_prefix(payload))

    def health_check(self) -> dict[str, Any]:
        self._wait_for_turn()
        response = self._session.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
        return {"ok": response.ok, "status_code": response.status_code, "provider": self.provider_name}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._wait_for_turn()
        try:
            response = self._session.post(
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.Timeout as exc:
            raise self._timeout_error(exc) from exc
        response.encoding = "utf-8"
        if not response.ok:
            raise self._provider_error(response)
        return response.json()

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
