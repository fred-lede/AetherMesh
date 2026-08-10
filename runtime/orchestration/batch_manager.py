from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from config.settings import settings

logger = logging.getLogger("runtime.orchestration.batch_manager")

ALLOWED_BATCH_ENDPOINTS = {"/v1/chat/completions", "/v1/responses", "/v1/embeddings"}
COMPLETION_WINDOWS = {"24h", "48h", "1h"}


def _now() -> int:
    return int(time.time())


def _parse_completion_window(window: str) -> int:
    value = str(window).strip().lower()
    if value in COMPLETION_WINDOWS:
        return _now() + {"1h": 3600, "24h": 86400, "48h": 172800}[value]
    if value.endswith("h") and value[:-1].isdigit():
        return _now() + int(value[:-1]) * 3600
    return _now() + 86400


class BatchManager:
    def __init__(self, base_dir: Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else Path(settings.upload_dir) / "batches"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._batches: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for path in self._base_dir.glob("batch_*.json"):
            try:
                batch = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(batch, dict) and batch.get("id"):
                    self._batches[batch["id"]] = batch
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable batch file %s", path.name)

    def _save(self, batch: dict[str, Any]) -> None:
        path = self._base_dir / f"batch_{batch['id']}.json"
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        if endpoint not in ALLOWED_BATCH_ENDPOINTS:
            raise ValueError(f"Unsupported batch endpoint '{endpoint}'")
        source = Path(settings.upload_dir) / input_file_id
        if not source.exists():
            raise ValueError(f"Input file '{input_file_id}' not found")

        batch_id = f"batch_{uuid.uuid4().hex[:16]}"
        batch = {
            "id": batch_id,
            "object": "batch",
            "endpoint": endpoint,
            "input_file_id": input_file_id,
            "completion_window": str(completion_window),
            "status": "validating",
            "created_at": _now(),
            "expires_at": _parse_completion_window(completion_window),
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "errors": [],
        }
        if user_id:
            batch["user_id"] = user_id

        try:
            items = self._load_input_file(source)
        except ValueError as exc:
            batch["status"] = "failed"
            batch["error"] = str(exc)
            batch["completed_at"] = _now()
            with self._lock:
                self._batches[batch_id] = batch
                self._save(batch)
            return batch

        batch["request_counts"]["total"] = len(items)
        with self._lock:
            self._batches[batch_id] = batch
            self._save(batch)

        thread = threading.Thread(
            target=self._process,
            args=(batch_id, items, handler),
            name=f"batch-{batch_id}",
            daemon=True,
        )
        thread.start()
        return batch

    @staticmethod
    def _load_input_file(source: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
            if not isinstance(item, dict) or "custom_id" not in item:
                raise ValueError(f"Line {line_no} missing required 'custom_id'")
            item.setdefault("method", "POST")
            item.setdefault("url", "/v1/chat/completions")
            if item["method"] not in {"POST", "GET"}:
                raise ValueError(f"Line {line_no} has unsupported method '{item['method']}'")
            if "body" not in item:
                raise ValueError(f"Line {line_no} missing required 'body'")
            items.append(item)
        if not items:
            raise ValueError("Input file contains no batch requests")
        return items

    def _process(self, batch_id: str, items: list[dict[str, Any]], handler: Callable) -> None:
        with self._lock:
            batch = self._batches.get(batch_id)
        if batch is None:
            return
        batch["status"] = "in_progress"
        self._save(batch)

        output_lines: list[dict[str, Any]] = []
        for item in items:
            request_id = f"req_{uuid.uuid4().hex[:24]}"
            if batch["status"] == "cancelling":
                batch["status"] = "cancelled"
                self._save(batch)
                return
            try:
                body = handler(item.get("body") or {})
                output_lines.append(
                    {
                        "id": request_id,
                        "custom_id": item["custom_id"],
                        "response": {"status_code": 200, "request_id": request_id, "body": body},
                        "error": None,
                    }
                )
                batch["request_counts"]["completed"] += 1
            except Exception as exc:
                output_lines.append(
                    {
                        "id": None,
                        "custom_id": item["custom_id"],
                        "response": None,
                        "error": {"message": str(exc), "code": "batch_request_failed", "param": None},
                    }
                )
                batch["request_counts"]["failed"] += 1
                batch["errors"].append({"custom_id": item["custom_id"], "message": str(exc)})
            if (batch["request_counts"]["completed"] + batch["request_counts"]["failed"]) % 10 == 0:
                self._save(batch)

        output_file_id = f"file_batch_out_{batch_id}"
        output_path = Path(settings.upload_dir) / output_file_id
        output_path.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in output_lines) + "\n",
            encoding="utf-8",
        )

        if batch["status"] == "cancelling":
            batch["status"] = "cancelled"
        else:
            batch["status"] = "completed"
        batch["output_file_id"] = output_file_id
        batch["completed_at"] = _now()
        self._save(batch)
        logger.info("Batch %s completed: total=%d ok=%d failed=%d",
            batch_id,
            batch["request_counts"]["total"],
            batch["request_counts"]["completed"],
            batch["request_counts"]["failed"],
        )

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        return self._batches.get(batch_id)

    def list_batches(self, limit: int = 20, after: str | None = None) -> list[dict[str, Any]]:
        batches = sorted(self._batches.values(), key=lambda b: b.get("created_at", 0), reverse=True)
        if after:
            batches = [b for b in batches if b["id"] > after]
        return batches[: max(1, min(limit, 100))]

    def cancel_batch(self, batch_id: str) -> dict[str, Any] | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        if batch["status"] in {"in_progress", "validating"}:
            batch["status"] = "cancelling"
            self._save(batch)
        return batch

    def _public_batch(self, batch_id: str) -> dict[str, Any] | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        public = {k: v for k, v in batch.items() if k != "errors"}
        return public


batch_manager = BatchManager()
