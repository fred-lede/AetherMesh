from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from providers.base import ProviderAdapter, ProviderError


@dataclass
class Credential:
    api_key: str
    base_url: str | None = None
    label: str = ""
    cooldown_s: int | None = None


class CredentialPool(ProviderAdapter):
    """Wraps a ProviderAdapter subclass with multi-key failover.

    On rate-limit (429), auth (401/403), or other retryable provider errors,
    the failed credential is placed on cooldown and the next available
    credential is transparently retried.

    Stream retry is best-effort: retries only if the initial POST fails
    before any data is yielded. Mid-stream failures propagate to the caller.
    """

    RETRYABLE_STATUSES = frozenset({401, 403, 429, 502, 503, 504})
    provider_name: str = "credential_pool"

    def __init__(
        self,
        adapter_cls: type[ProviderAdapter],
        credentials: list[Credential],
        default_cooldown_s: int = 300,
    ) -> None:
        if not credentials:
            raise ProviderError("CredentialPool: no credentials provided")
        self._adapter_cls = adapter_cls
        self._credentials = list(credentials)
        self._default_cooldown_s = default_cooldown_s
        self._cooldowns: dict[str, float] = {}
        self._index = 0
        self._lock = threading.Lock()
        self._current_stream_adapter: ProviderAdapter | None = None

    # ── public API ──────────────────────────────────────────────

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._retry("chat", payload)

    def responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._retry("responses", payload)

    def stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any] | str]:
        last_error: ProviderError | None = None
        for _ in range(self._max_attempts()):
            cred = self._next_credential()
            if cred is None:
                raise last_error or ProviderError(
                    "All credentials are in cooldown; none available",
                    status_code=429, code="provider_rate_limited",
                )
            adapter = self._build(cred)
            self._current_stream_adapter = adapter
            try:
                yield from adapter.stream(payload)
                return
            except ProviderError as exc:
                if self._is_retryable(exc):
                    self._mark_cooldown(cred, exc.retry_after)
                    last_error = exc
                    continue
                raise
        raise last_error or ProviderError(
            "CredentialPool: stream exhausted all retries",
            status_code=503, code="provider_overloaded",
        )

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._retry("embeddings", payload)

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._retry("rerank", payload)

    def health_check(self) -> dict[str, Any]:
        result = {"ok": False, "provider": self._adapter_cls.provider_name}
        for cred in self._credentials:
            adapter = self._build(cred)
            try:
                return adapter.health_check()
            except Exception:
                continue
        return result

    def abort_stream(self) -> None:
        adapter = self._current_stream_adapter
        if adapter is not None:
            adapter.abort_stream()

    def get_cooldown_status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        status: list[dict[str, Any]] = []
        for cred in self._credentials:
            remaining = max(0.0, self._cooldowns.get(cred.api_key, 0.0) - now)
            status.append({
                "api_key": cred.api_key[:8] + "..." if len(cred.api_key) > 12 else cred.api_key,
                "label": cred.label,
                "on_cooldown": remaining > 0,
                "cooldown_remaining_s": round(remaining, 1),
            })
        return status

    # ── internal helpers ────────────────────────────────────────

    def _retry(self, method: str, payload: dict[str, Any]) -> Any:
        last_error: ProviderError | None = None
        for _ in range(self._max_attempts()):
            cred = self._next_credential()
            if cred is None:
                raise last_error or ProviderError(
                    "All credentials are in cooldown; none available",
                    status_code=429, code="provider_rate_limited",
                )
            adapter = self._build(cred)
            try:
                fn = getattr(adapter, method)
                return fn(payload)
            except ProviderError as exc:
                if self._is_retryable(exc):
                    self._mark_cooldown(cred, exc.retry_after)
                    last_error = exc
                    continue
                raise
        raise last_error or ProviderError(
            f"CredentialPool: {method} exhausted all retries",
            status_code=503, code="provider_overloaded",
        )

    def _build(self, cred: Credential) -> ProviderAdapter:
        return self._adapter_cls(api_key=cred.api_key, base_url=cred.base_url)

    def _next_credential(self) -> Credential | None:
        now = time.monotonic()
        with self._lock:
            for _ in range(len(self._credentials)):
                self._index = (self._index + 1) % len(self._credentials)
                candidate = self._credentials[self._index]
                if now >= self._cooldowns.get(candidate.api_key, 0.0):
                    return candidate
            return None

    def _mark_cooldown(self, cred: Credential, retry_after: int | None = None) -> None:
        delay = min(
            max(retry_after or cred.cooldown_s or self._default_cooldown_s, 5),
            3600,
        )
        with self._lock:
            self._cooldowns[cred.api_key] = time.monotonic() + delay

    def _is_retryable(self, exc: ProviderError) -> bool:
        return exc.status_code in self.RETRYABLE_STATUSES

    def _max_attempts(self) -> int:
        return max(len(self._credentials) * 2, 3)
