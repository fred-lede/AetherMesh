from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.document_router as document_module
from runtime.documents.mineru_converter import MinerUError
from runtime.security.middleware import add_security_middleware


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(document_module.router)
    add_security_middleware(app, enable_auth=False, enable_rate_limit=False)
    return TestClient(app)


def test_document_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(document_module, "mineru_available", lambda: True)
    client = _make_client()
    r = client.get("/v1/documents/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["mineru_available"] is True


def test_document_extract_upload_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_convert(path, out_dir=None, backend=None, method=None, timeout_s=None, include_images=False):
        return {
            "markdown": "# extracted title\n\nbody text",
            "source": str(path),
            "output_path": str(tmp_path / "out.md"),
            "duration_ms": 12,
            "chars": 22,
            "image_count": 0,
            "images": [],
        }

    monkeypatch.setattr(document_module, "convert_document", fake_convert)
    client = _make_client()
    r = client.post(
        "/v1/documents/extract",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["filename"] == "doc.pdf"
    assert "extracted title" in data["markdown"]
    assert data["chars"] == 22


def test_document_extract_upload_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(document_module, "convert_document", lambda *a, **k: {})
    client = _make_client()
    r = client.post("/v1/documents/extract", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_document_extract_upload_mineru_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_err(*a, **k):
        raise MinerUError("MinerU exited with code 1")

    monkeypatch.setattr(document_module, "convert_document", raise_err)
    client = _make_client()
    r = client.post(
        "/v1/documents/extract",
        files={"file": ("bad.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 422
    assert r.json()["ok"] is False
    assert "code 1" in r.json()["error"]


def test_document_extract_json_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_convert(path, out_dir=None, backend=None, method=None, timeout_s=None, include_images=False):
        captured["path"] = str(path)
        captured["out_dir"] = out_dir
        return {
            "markdown": "## md",
            "source": str(path),
            "output_path": str(tmp_path / "x.md"),
            "duration_ms": 5,
            "chars": 5,
            "image_count": 0,
            "images": [],
        }

    monkeypatch.setattr(document_module, "convert_document", fake_convert)
    client = _make_client()
    r = client.post(
        "/v1/documents/extract/json",
        json={"path": "/data/report.pdf", "backend": "pipeline"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert captured["path"] == "/data/report.pdf"


def test_document_extract_json_missing_path() -> None:
    client = _make_client()
    r = client.post("/v1/documents/extract/json", json={"backend": "pipeline"})
    assert r.status_code == 400


def test_document_extract_async_returns_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_job_id = "abc123"

    class _FakeJobs:
        def __init__(self) -> None:
            self.submitted: dict = {}

        def submit(self, data, filename, include_images=False, backend=None, method=None, timeout_s=None):
            self.submitted = {
                "data": data,
                "filename": filename,
                "include_images": include_images,
                "backend": backend,
                "method": method,
                "timeout_s": timeout_s,
            }
            return fake_job_id

        def get(self, job_id):
            return {"id": job_id, "status": "queued", "filename": "doc.pdf", "error": None, "result": None,
                    "created_at": None, "started_at": None, "finished_at": None}

        def list(self, limit=20):
            return []

    fake = _FakeJobs()
    monkeypatch.setattr(document_module, "_job_manager", fake)
    client = _make_client()
    r = client.post(
        "/v1/documents/extract/async",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["job_id"] == fake_job_id
    assert data["status"] == "queued"
    assert fake.submitted["filename"] == "doc.pdf"
    assert fake.submitted["include_images"] is False


def test_document_extract_async_empty() -> None:
    client = _make_client()
    r = client.post("/v1/documents/extract/async", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_document_job_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "abc": {"id": "abc", "status": "completed", "filename": "doc.pdf", "error": None,
                "result": {"markdown": "## done", "chars": 8, "image_count": 0},
                "created_at": None, "started_at": None, "finished_at": None}
    }

    class _FakeJobs:
        def submit(self, *a, **k):
            return "abc"

        def get(self, job_id):
            return fake.get(job_id)

        def list(self, limit=20):
            return list(fake.values())

    monkeypatch.setattr(document_module, "_job_manager", _FakeJobs())
    client = _make_client()
    r = client.get("/v1/documents/jobs/abc")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["status"] == "completed"
    assert data["result"]["markdown"] == "## done"


def test_document_job_status_not_found() -> None:
    client = _make_client()
    r = client.get("/v1/documents/jobs/nope")
    assert r.status_code == 404


def test_document_job_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeJobs:
        def submit(self, *a, **k):
            return "abc"

        def get(self, job_id):
            return None

        def list(self, limit=20):
            return [{"id": "abc", "status": "queued"}]

    monkeypatch.setattr(document_module, "_job_manager", _FakeJobs())
    client = _make_client()
    r = client.get("/v1/documents/jobs")
    assert r.status_code == 200
    assert r.json()["jobs"][0]["id"] == "abc"


def test_document_extract_json_async(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    fake_job_id = "jsonjob1"

    class _FakeJobs:
        def __init__(self) -> None:
            self.submitted: dict = {}

        def submit(self, data, filename, include_images=False, backend=None, method=None, timeout_s=None):
            self.submitted = {
                "data": data,
                "filename": filename,
                "include_images": include_images,
                "backend": backend,
                "method": method,
                "timeout_s": timeout_s,
            }
            return fake_job_id

        def get(self, job_id):
            return {"id": job_id, "status": "queued", "filename": "doc.pdf", "error": None, "result": None,
                    "created_at": None, "started_at": None, "finished_at": None}

        def list(self, limit=20):
            return []

    fake = _FakeJobs()
    monkeypatch.setattr(document_module, "_job_manager", fake)
    client = _make_client()
    r = client.post(
        "/v1/documents/extract/json/async",
        json={"path": str(pdf), "include_images": True, "backend": "pipeline", "timeout_s": 900},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == fake_job_id
    assert fake.submitted["filename"] == "doc.pdf"
    assert fake.submitted["include_images"] is True
    assert fake.submitted["backend"] == "pipeline"
    assert fake.submitted["timeout_s"] == 900


def test_document_extract_json_async_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeJobs:
        def submit(self, *a, **k):
            return "x"

    monkeypatch.setattr(document_module, "_job_manager", _FakeJobs())
    client = _make_client()
    r = client.post("/v1/documents/extract/json/async", json={"path": "C:\\nope\\missing.pdf"})
    assert r.status_code == 404
