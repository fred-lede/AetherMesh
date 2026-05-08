"""
Circuit Breaker implementation for AetherMesh.

States: CLOSED → OPEN → HALF_OPEN → CLOSED

When a service fails repeatedly, circuit opens to prevent cascading failures.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable

LOGGER = logging.getLogger("aiih.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half"    # Testing if recovery


class CircuitBreaker:
    """
    Circuit breaker for worker/provider calls.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Successes needed to close circuit
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    LOGGER.info(f"Circuit {self.name}: OPEN -> HALF_OPEN (testing recovery)")
            return self._state

    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    LOGGER.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, go back to OPEN
                self._state = CircuitState.OPEN
                LOGGER.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (recovery test failed)")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    LOGGER.warning(f"Circuit {self.name}: CLOSED -> OPEN (failure threshold reached)")

    def call(self, func: Callable[[], Any], fallback: Any = None) -> Any:
        """
        Execute func with circuit breaker protection.
        
        Args:
            func: Function to call
            fallback: Value to return if circuit is OPEN
            
        Returns:
            Result of func or fallback
        """
        if not self.is_available():
            LOGGER.warning(f"Circuit {self.name}: OPEN, rejecting request")
            if callable(fallback):
                return fallback()
            return fallback

        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def get_state(self) -> dict[str, Any]:
        """Get current state for monitoring."""
        return {
            "circuit": self.name,
            "state": self.state.value,
            "failures": self._failure_count,
            "successes": self._success_count,
            "last_failure": self._last_failure_time,
        }


class CircuitBreakerRegistry:
    """Registry of circuit breakers for all workers/providers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        name: str,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
            return self._breakers[name]

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get states of all breakers."""
        with self._lock:
            return {name: cb.get_state() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuits to CLOSED."""
        with self._lock:
            for cb in self._breakers.values():
                cb._state = CircuitState.CLOSED
                cb._failure_count = 0
                cb._success_count = 0
        LOGGER.info("All circuits reset to CLOSED")


# Singleton instance
_circuit_registry: CircuitBreakerRegistry | None = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get or create the circuit breaker registry singleton."""
    global _circuit_registry
    if _circuit_registry is None:
        _circuit_registry = CircuitBreakerRegistry()
    return _circuit_registry