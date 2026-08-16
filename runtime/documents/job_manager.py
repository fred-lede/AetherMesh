from __future__ import annotations

import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.documents.mineru_converter import convert_document

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"


class DocumentJobManager:
    def __init__(self, ttl_s: int = 3600, sweep_s: int = 60, autostart: bool = True) -> None:
        self._ttl_s = ttl_s
        self._sweep_s = sweep_s
        self._autostart = autostart
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._sweeper: threading.Thread | None = None

    def _ensure_threads(self) -> None:
        if not self._autostart:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop, name="document-jobs", daemon=True)
            self._worker.start()
        if self._sweeper is None or not self._sweeper.is_alive():
            self._sweeper = threading.Thread(target=self._sweep_loop, name="document-jobs-sweep", daemon=True)
            self._sweeper.start()

    def submit(
        self,
        file_bytes: bytes,
        filename: str,
        include_images: bool = False,
        backend: str | None = None,
        method: str | None = None,
        timeout_s: int | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        job_dir = Path(tempfile.mkdtemp(prefix="aethermesh_job_"))
        file_path = job_dir / filename
        file_path.write_bytes(file_bytes)
        now = _now()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": QUEUED,
                "filename": filename,
                "error": None,
                "result": None,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "file_path": str(file_path),
                "out_dir": str(job_dir),
                "include_images": include_images,
                "backend": backend,
                "method": method,
                "timeout_s": timeout_s,
            }
        self._ensure_threads()
        self._queue.put(job_id)
        return job_id
    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _public(job) if job else None

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return [_public(j) for j in jobs[:limit]]

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            except Exception as exc:  # noqa: BLE001
                self._fail(job_id, f"{type(exc).__name__}: {exc}")

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = PROCESSING
            job["started_at"] = _now()
        try:
            result = convert_document(
                job["file_path"],
                out_dir=job["out_dir"],
                backend=job["backend"],
                method=job["method"],
                timeout_s=job["timeout_s"],
                include_images=job["include_images"],
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(job_id, f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = COMPLETED
            job["result"] = result
            job["finished_at"] = _now()

    def _fail(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = FAILED
            job["error"] = message
            job["finished_at"] = _now()

    def _sweep_loop(self) -> None:
        while True:
            time.sleep(self._sweep_s)
            self._sweep()

    def _sweep(self) -> None:
        cutoff = time.time() - self._ttl_s
        expired: list[tuple[str, str]] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                finished = job.get("finished_at")
                if finished and finished < cutoff:
                    expired.append((job_id, job["out_dir"]))
                    del self._jobs[job_id]
        for _job_id, out_dir in expired:
            try:
                import shutil

                shutil.rmtree(out_dir, ignore_errors=True)
            except OSError:
                pass


def _now() -> float:
    return time.time()


def _public(job: dict[str, Any]) -> dict[str, Any]:
    def _iso(ts: float | None) -> str | None:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    return {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "error": job.get("error"),
        "result": job.get("result"),
        "created_at": _iso(job["created_at"]),
        "started_at": _iso(job.get("started_at")),
        "finished_at": _iso(job.get("finished_at")),
    }
