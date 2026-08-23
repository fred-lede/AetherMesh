from __future__ import annotations

import time
from collections import defaultdict


class TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.time()

    def refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    _SWEEP_INTERVAL_S = 60.0
    _IDLE_TTL_S = 3600.0

    def __init__(self, default_rate: float = 10.0, default_burst: int = 20) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self.default_rate = default_rate
        self.default_burst = default_burst
        self._last_sweep = time.time()

    def _sweep_idle(self) -> None:
        now = time.time()
        if now - self._last_sweep < self._SWEEP_INTERVAL_S:
            return
        self._last_sweep = now
        cutoff = now - self._IDLE_TTL_S
        stale = [key for key, bucket in self._buckets.items() if bucket.last_refill < cutoff]
        for key in stale:
            del self._buckets[key]

    def _get_bucket(self, key: str) -> TokenBucket:
        self._sweep_idle()
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.default_rate, self.default_burst)
        return self._buckets[key]

    def check(self, key: str, tokens: int = 1) -> bool:
        return self._get_bucket(key).consume(tokens)

    def get_remaining(self, key: str) -> float:
        bucket = self._get_bucket(key)
        bucket.refill()
        return bucket.tokens

    def set_limit(self, key: str, rate: float, burst: int) -> None:
        self._buckets[key] = TokenBucket(rate, burst)

    def reset(self, key: str) -> None:
        self._buckets.pop(key, None)

    def clear(self) -> None:
        self._buckets.clear()


rate_limiter = RateLimiter()
