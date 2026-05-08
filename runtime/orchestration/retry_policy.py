from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("orchestration.retry")


@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_multiplier: float = 2.0
    retryable_errors: tuple[str, ...] = ("timeout", "rate_limit", "overloaded", "unavailable")

    async def execute(
        self,
        handler: Callable[[], Any],
        node_id: str = "",
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as exc:
                last_error = exc
                exc_name = type(exc).__name__.lower()
                if not any(err in exc_name for err in self.retryable_errors):
                    if attempt >= self.max_retries:
                        raise
                if attempt >= self.max_retries:
                    raise
                wait = self.backoff_base_s * (self.backoff_multiplier ** attempt)
                logger.info(
                    "Retry %s/%s for node %s after %.1fs: %s",
                    attempt + 1, self.max_retries, node_id, wait, exc,
                )
                await asyncio.sleep(wait)
        raise last_error  # type: ignore[misc]
