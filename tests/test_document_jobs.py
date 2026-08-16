from __future__ import annotations

import time
from pathlib import Path

import pytest

from runtime.documents import job_manager
from runtime.documents.job_manager import (
    COMPLETED,
    FAILED,
    PROCESSING,
    QUEUED,
    DocumentJobManager,
)


def _public_status(job: dict | None) -> str | None:
    return job["status"] if job else None


def test_submit_returns_queued_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    def fake_convert(*args, **kwargs):
        calls["kwargs"] = kwargs
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=1, sweep_s=1)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf", include_images=True, backend="pipeline", timeout_s=900)
    assert job_id
    job = mgr.get(job_id)
    assert _public_status(job) == QUEUED
    assert job["filename"] == "doc.pdf"


def test_execute_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_convert(*args, **kwargs):
        return {"markdown": "# hello", "chars": 8, "image_count": 0}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")
    mgr._execute(job_id)
    job = mgr.get(job_id)
    assert _public_status(job) == COMPLETED
    assert job["result"]["markdown"] == "# hello"


def test_execute_failure_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_convert(*args, **kwargs):
        raise RuntimeError("MinerU exploded")

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")
    mgr._execute(job_id)
    job = mgr.get(job_id)
    assert _public_status(job) == FAILED
    assert "MinerU exploded" in job["error"]


def test_missing_job_returns_none() -> None:
    mgr = DocumentJobManager()
    assert mgr.get("nope") is None


def test_list_orders_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_convert(*args, **kwargs):
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    first = mgr.submit(b"a", "a.pdf")
    time.sleep(0.01)
    second = mgr.submit(b"b", "b.pdf")
    jobs = mgr.list()
    assert [j["id"] for j in jobs] == [second, first]


def test_processing_status_during_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    gate: dict = {"entered": False, "release": False}

    def fake_convert(*args, **kwargs):
        gate["entered"] = True
        while not gate["release"]:
            time.sleep(0.005)
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")

    import threading

    t = threading.Thread(target=mgr._execute, args=(job_id,))
    t.start()
    while not gate["entered"]:
        time.sleep(0.005)
    assert _public_status(mgr.get(job_id)) == PROCESSING
    gate["release"] = True
    t.join(timeout=5)
    assert _public_status(mgr.get(job_id)) == COMPLETED


def test_sweep_removes_expired_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_convert(*args, **kwargs):
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")
    mgr._execute(job_id)
    with mgr._lock:
        mgr._jobs[job_id]["finished_at"] = time.time() - 7200
    mgr._sweep()
    assert mgr.get(job_id) is None


def test_public_omits_internal_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_convert(*args, **kwargs):
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(ttl_s=3600, sweep_s=3600)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")
    job = mgr.get(job_id)
    assert "file_path" not in job
    assert "out_dir" not in job
    assert "id" in job
    assert "created_at" in job


def test_public_has_iso_timestamps() -> None:
    job = job_manager._public({
        "id": "x", "status": COMPLETED, "filename": "f.pdf", "error": None, "result": None,
        "created_at": 1000.0, "started_at": 1001.0, "finished_at": 1002.0,
    })
    assert job["created_at"].startswith("1970-01-01T")
    assert job["started_at"].startswith("1970-01-01T")
    assert job["finished_at"].startswith("1970-01-01T")


def test_background_worker_processes_job(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_convert(*args, **kwargs):
        return {"markdown": "# bg done", "chars": 9, "image_count": 0}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(autostart=True)
    job_id = mgr.submit(b"%PDF-1.4", "doc.pdf")
    for _ in range(100):
        time.sleep(0.02)
        if _public_status(mgr.get(job_id)) == COMPLETED:
            break
    job = mgr.get(job_id)
    assert _public_status(job) == COMPLETED
    assert job["result"]["markdown"] == "# bg done"


def test_background_worker_serializes_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    peak = 0
    started: list[int] = []

    def fake_convert(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.append(active)
        time.sleep(0.01)
        active -= 1
        return {"markdown": "md", "chars": 2}

    monkeypatch.setattr(job_manager, "convert_document", fake_convert)
    mgr = DocumentJobManager(autostart=True)
    ids = [mgr.submit(b"a", f"d{i}.pdf") for i in range(5)]
    for _ in range(200):
        time.sleep(0.02)
        if all(_public_status(mgr.get(j)) == COMPLETED for j in ids):
            break
    assert peak == 1
    assert len(started) == 5
