from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_total = 0
        self._request_latency_sum_ms = 0.0
        self._request_latency_count = 0
        self._error_count = 0
        self._endpoint_usage: dict[str, int] = defaultdict(int)
        self._model_usage: dict[str, int] = defaultdict(int)
        self._provider_status: dict[str, int] = {}
        self._worker_usage: dict[str, dict[str, Any]] = {}
        self._queue_length = 0
        self._recent_events: deque[dict[str, Any]] = deque()
        self._latency_recent_ms: deque[float] = deque(maxlen=5000)
        self._latency_by_endpoint_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2000))
        self._latency_by_endpoint_model_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._error_code_usage: dict[str, int] = defaultdict(int)
        self._request_total_history: deque[float] = deque(maxlen=50)

    def _prune_recent_events(self, now: float, keep_seconds: int = 3600) -> None:
        cutoff = now - max(keep_seconds, 300)
        while self._recent_events and float(self._recent_events[0].get("ts", 0.0)) < cutoff:
            self._recent_events.popleft()

    def _rolling_error_rate(
        self,
        *,
        now: float,
        window_s: int,
        model_filter: str | None = None,
    ) -> dict[str, Any]:
        window_start = now - max(1, window_s)
        requests = 0
        errors = 0
        for event in self._recent_events:
            ts = float(event.get("ts", 0.0))
            if ts < window_start:
                continue
            model = str(event.get("model", ""))
            if model_filter and model != model_filter:
                continue
            requests += 1
            if bool(event.get("error", False)):
                errors += 1
        error_rate = (errors / requests) if requests else 0.0
        return {
            "window_s": window_s,
            "requests": requests,
            "errors": errors,
            "error_rate": round(error_rate, 4),
        }

    def _percentile(self, samples: deque[float], p: float) -> float:
        if not samples:
            return 0.0
        sorted_values = sorted(float(v) for v in samples)
        index = int(round((len(sorted_values) - 1) * p))
        index = max(0, min(index, len(sorted_values) - 1))
        return float(sorted_values[index])

    def record_request(
        self,
        *,
        endpoint: str,
        latency_ms: float,
        model: str = "",
        worker_id: str = "",
        provider: str = "",
        error: bool = False,
        error_code: str = "",
    ) -> None:
        with self._lock:
            now = time.time()
            latency_value = max(latency_ms, 0.0)
            self._request_total += 1
            self._request_latency_sum_ms += latency_value
            self._request_latency_count += 1
            self._endpoint_usage[endpoint] += 1
            if model:
                self._model_usage[model] += 1
            if error:
                self._error_count += 1
            if error_code:
                self._error_code_usage[str(error_code)] += 1

            self._request_total_history.append(self._request_total)
            self._latency_recent_ms.append(latency_value)
            self._latency_by_endpoint_ms[endpoint].append(latency_value)
            if model:
                self._latency_by_endpoint_model_ms[f"{endpoint}|{model}"].append(latency_value)

            self._recent_events.append(
                {
                    "ts": now,
                    "endpoint": endpoint,
                    "model": model,
                    "worker_id": worker_id,
                    "provider": provider,
                    "error": bool(error),
                    "error_code": str(error_code or ""),
                }
            )
            self._prune_recent_events(now)

    def set_worker_usage(
        self,
        worker_id: str,
        *,
        node_id: str,
        gpu_utilization: float,
        queue_size: int,
        status: str,
    ) -> None:
        with self._lock:
            self._worker_usage[worker_id] = {
                "node_id": node_id,
                "gpu_utilization": gpu_utilization,
                "queue_size": queue_size,
                "status": status,
            }

    def set_queue_length(self, queue_length: int) -> None:
        with self._lock:
            self._queue_length = max(queue_length, 0)

    def set_provider_status(self, provider: str, healthy: bool) -> None:
        with self._lock:
            self._provider_status[provider] = int(healthy)

    def increment_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            self._prune_recent_events(now)

            avg_latency = 0.0
            if self._request_latency_count:
                avg_latency = self._request_latency_sum_ms / self._request_latency_count

            p95 = self._percentile(self._latency_recent_ms, 0.95)
            p99 = self._percentile(self._latency_recent_ms, 0.99)

            endpoint_latency: dict[str, dict[str, Any]] = {}
            for endpoint, samples in self._latency_by_endpoint_ms.items():
                endpoint_latency[endpoint] = {
                    "samples": len(samples),
                    "p95": round(self._percentile(samples, 0.95), 3),
                    "p99": round(self._percentile(samples, 0.99), 3),
                }

            endpoint_model_latency: dict[str, dict[str, Any]] = {}
            for key, samples in self._latency_by_endpoint_model_ms.items():
                endpoint_model_latency[key] = {
                    "samples": len(samples),
                    "p95": round(self._percentile(samples, 0.95), 3),
                    "p99": round(self._percentile(samples, 0.99), 3),
                }

            vision_model_stats_5m = self._rolling_error_rate(now=now, window_s=300, model_filter="qwen3-vl:8b")
            vision_endpoint_stats_5m = self._rolling_error_rate(now=now, window_s=300)

            return {
                "request_total": self._request_total,
                "request_latency_ms_avg": round(avg_latency, 3),
                "request_latency_ms_p95": round(p95, 3),
                "request_latency_ms_p99": round(p99, 3),
                "request_latency_ms_sum": round(self._request_latency_sum_ms, 3),
                "request_latency_samples": self._request_latency_count,
                "endpoint_latency": endpoint_latency,
                "endpoint_model_latency": endpoint_model_latency,
                "worker_usage": dict(self._worker_usage),
                "queue_length": self._queue_length,
                "model_usage": dict(self._model_usage),
                "endpoint_usage": dict[str, int](self._endpoint_usage),
                "provider_status": dict(self._provider_status),
                "error_count": self._error_count,
                "error_code_usage": dict(self._error_code_usage),
                "recent_events": list(self._recent_events),
                "vision_error_rate_5m": {
                    "model": vision_model_stats_5m,
                    "all_requests": vision_endpoint_stats_5m,
                },
            }

def render_prometheus_text(snapshot: dict[str, Any]) -> str:
    lines = [
        "# HELP request_total Total routed inference requests.",
        "# TYPE request_total counter",
        f"request_total {snapshot.get('request_total', 0)}",
        "# HELP request_latency Average request latency in milliseconds.",
        "# TYPE request_latency gauge",
        f"request_latency {snapshot.get('request_latency_ms_avg', 0.0)}",
        "# HELP request_latency_p95 95th percentile request latency in milliseconds.",
        "# TYPE request_latency_p95 gauge",
        f"request_latency_p95 {snapshot.get('request_latency_ms_p95', 0.0)}",
        "# HELP request_latency_p99 99th percentile request latency in milliseconds.",
        "# TYPE request_latency_p99 gauge",
        f"request_latency_p99 {snapshot.get('request_latency_ms_p99', 0.0)}",
        "# HELP queue_length Pending async tasks in Redis.",
        "# TYPE queue_length gauge",
        f"queue_length {snapshot.get('queue_length', 0)}",
        "# HELP error_count Total request processing errors.",
        "# TYPE error_count counter",
        f"error_count {snapshot.get('error_count', 0)}",
    ]

    vision_5m = snapshot.get("vision_error_rate_5m", {}).get("model", {})
    lines.append("# HELP vision_error_rate_5m Vision model error rate in last 5 minutes.")
    lines.append("# TYPE vision_error_rate_5m gauge")
    lines.append(f"vision_error_rate_5m {vision_5m.get('error_rate', 0.0)}")

    for model, count in sorted(snapshot.get("model_usage", {}).items()):
        lines.append(f'model_usage{{model="{model}"}} {count}')

    for code, count in sorted(snapshot.get("error_code_usage", {}).items()):
        lines.append(f'error_code_count{{code="{code}"}} {count}')

    for worker_id, details in sorted(snapshot.get("worker_usage", {}).items()):
        node_id = details.get("node_id", "unknown")
        status = details.get("status", "unknown")
        gpu_utilization = float(details.get("gpu_utilization", 0.0))
        queue_size = int(details.get("queue_size", 0))
        lines.append(
            f'worker_usage{{worker_id="{worker_id}",node_id="{node_id}",status="{status}",metric="gpu_utilization"}} {gpu_utilization}'
        )
        lines.append(
            f'worker_usage{{worker_id="{worker_id}",node_id="{node_id}",status="{status}",metric="queue_size"}} {queue_size}'
        )

    for provider, healthy in sorted(snapshot.get("provider_status", {}).items()):
        lines.append(f'provider_status{{provider="{provider}"}} {healthy}')

    return "\n".join(lines) + "\n"

metrics_store = MetricsStore()
