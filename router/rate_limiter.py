"""
Rate limiting middleware.
Simple token bucket implementation per IP address.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from config.settings import settings

LOGGER = logging.getLogger("aiih.ratelimit")


class TokenBucket:
    """Token bucket for rate limiting."""

    __slots__ = ("tokens", "last_refill", "lock")

    def __init__(self, capacity: int):
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            refill_rate = 1.0  # tokens per second (will be multiplied by rate_limit_per_minute)
            self.tokens = min(
                self.tokens + elapsed * refill_rate,
                settings.rate_limit_burst,
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimiter:
    """Rate limiter using token bucket per IP."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(settings.rate_limit_burst)
        )
        self._lock = threading.Lock()
        self._cleanup_interval = 3600  # seconds
        self._last_cleanup = time.time()

    def _get_bucket(self, key: str) -> TokenBucket:
        with self._lock:
            # Periodic cleanup of old entries
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale()
                self._last_cleanup = now
            return self._buckets[key]

    def _cleanup_stale(self) -> None:
        """Remove buckets that haven't been used in 1 hour."""
        now = time.time()
        stale_keys = [
            k
            for k, v in self._buckets.items()
            if now - v.last_refill > 3600
        ]
        for k in stale_keys:
            del self._buckets[k]
        if stale_keys:
            LOGGER.info(f"Cleaned up {len(stale_keys)} stale rate limit buckets")

    def check(self, client_ip: str, limit_per_minute: int) -> None:
        """Check rate limit. Raises HTTPException if exceeded."""
        bucket = self._get_bucket(client_ip)

        # Convert limit_per_minute to tokens per second for the bucket
        refill_rate = limit_per_minute / 60.0
        bucket.tokens = min(
            bucket.tokens + (time.time() - bucket.last_refill) * refill_rate,
            settings.rate_limit_burst,
        )

        if not bucket.consume(1):
            retry_after = int((1 - bucket.tokens) / refill_rate) + 1
            LOGGER.warning(f"Rate limit exceeded for {client_ip}, retry after {retry_after}s")
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "rate_limit_exceeded",
                    "message": f"Rate limit: {limit_per_minute} requests per minute",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )


# Singleton instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header first (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    # Fall back to direct client IP
    client_ip = request.client.host if request.client else "unknown"
    return client_ip


async def rate_limit_middleware(request: Request) -> None:
    """FastAPI middleware for rate limiting."""
    if not settings.rate_limit_enabled:
        return

    # Skip rate limiting for health checks
    if request.url.path in {"/health", "/v1/models"}:
        return

    client_ip = get_client_ip(request)
    get_rate_limiter().check(client_ip, settings.rate_limit_per_minute)