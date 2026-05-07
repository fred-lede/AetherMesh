from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    model: str
    provider: str
    endpoint: str
    streaming: bool
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: bool = False
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)


class RequestMetricsCollector:
    """Tracks per-request metrics including tokens, latency, and provider routing."""

    def __init__(self, max_history: int = 500) -> None:
        self._lock = threading.RLock()
        self._max_history = max_history
        self._requests: deque[RequestRecord] = deque(maxlen=max_history)
        self._provider_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "errors": 0, "total_latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0}
        )
        self._model_provider_latency: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )
        self._hourly_windows: deque[dict[str, Any]] = deque(maxlen=24)

    def record_request(self, record: RequestRecord) -> None:
        with self._lock:
            self._requests.append(record)
            ps = self._provider_stats[record.provider]
            ps["requests"] += 1
            ps["total_latency_ms"] += record.latency_ms
            ps["input_tokens"] += record.input_tokens
            ps["output_tokens"] += record.output_tokens
            if record.error:
                ps["errors"] += 1

            key = f"{record.model}|{record.provider}"
            self._model_provider_latency[key]["latency"].append(record.latency_ms)

    def get_provider_metrics(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for provider, stats in self._provider_stats.items():
                reqs = stats["requests"]
                result[provider] = {
                    "requests": reqs,
                    "errors": stats["errors"],
                    "error_rate": round(stats["errors"] / reqs, 4) if reqs else 0.0,
                    "avg_latency_ms": round(stats["total_latency_ms"] / reqs, 1) if reqs else 0.0,
                    "total_input_tokens": stats["input_tokens"],
                    "total_output_tokens": stats["output_tokens"],
                }
            return result

    def get_provider_diagnostics(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            diagnostics: dict[str, dict[str, Any]] = {}
            providers = {record.provider for record in self._requests if record.provider}
            for provider in providers:
                records = [record for record in self._requests if record.provider == provider]
                if not records:
                    continue

                last_record = records[-1]
                last_success = next((record for record in reversed(records) if not record.error), None)
                last_error = next((record for record in reversed(records) if record.error), None)
                consecutive_errors = 0
                for record in reversed(records):
                    if not record.error:
                        break
                    consecutive_errors += 1

                diagnostics[provider] = {
                    "last_seen_at": last_record.timestamp,
                    "last_seen_model": last_record.model,
                    "last_latency_ms": round(last_record.latency_ms, 1),
                    "last_success_at": last_success.timestamp if last_success else None,
                    "last_success_model": last_success.model if last_success else "",
                    "last_success_latency_ms": round(last_success.latency_ms, 1) if last_success else 0.0,
                    "last_error_at": last_error.timestamp if last_error else None,
                    "last_error_model": last_error.model if last_error else "",
                    "last_error_message": last_error.error_message if last_error else "",
                    "consecutive_errors": consecutive_errors,
                }
            return diagnostics

    def get_recent_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._requests)[-limit:]
            return [
                {
                    "request_id": r.request_id,
                    "model": r.model,
                    "provider": r.provider,
                    "endpoint": r.endpoint,
                    "streaming": r.streaming,
                    "latency_ms": round(r.latency_ms, 1),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp,
                }
                for r in items
            ]

    def get_provider_latency(self, model: str, provider: str, window: int = 100) -> dict[str, float]:
        with self._lock:
            key = f"{model}|{provider}"
            samples = list(self._model_provider_latency.get(key, {}).get("latency", []))[-window:]
            if not samples:
                return {}
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            return {
                "min": round(sorted_samples[0], 1),
                "max": round(sorted_samples[-1], 1),
                "avg": round(sum(sorted_samples) / n, 1),
                "p50": round(sorted_samples[int(n * 0.5)], 1),
                "p95": round(sorted_samples[int(n * 0.95)], 1),
                "p99": round(sorted_samples[int(n * 0.99)], 1),
            }

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_requests = len(self._requests)
            total_errors = sum(1 for r in self._requests if r.error)
            total_input_tokens = sum(r.input_tokens for r in self._requests)
            total_output_tokens = sum(r.output_tokens for r in self._requests)
            latencies = [r.latency_ms for r in self._requests if not r.error]
            streaming_count = sum(1 for r in self._requests if r.streaming)

            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                avg_lat = sum(sorted_lat) / n
                p50 = sorted_lat[int(n * 0.5)]
                p95 = sorted_lat[min(int(n * 0.95), n - 1)]
                p99 = sorted_lat[min(int(n * 0.99), n - 1)]
            else:
                avg_lat = p50 = p95 = p99 = 0.0

            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": round(total_errors / total_requests, 4) if total_requests else 0.0,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_latency_ms": round(avg_lat, 1),
                "p50_latency_ms": round(p50, 1),
                "p95_latency_ms": round(p95, 1),
                "p99_latency_ms": round(p99, 1),
                "streaming_requests": streaming_count,
                "non_streaming_requests": total_requests - streaming_count,
            }


request_metrics = RequestMetricsCollector()
