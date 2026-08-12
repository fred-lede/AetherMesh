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
    def fake_convert(path, out_dir=None, backend=None, method=None, timeout_s=None):
        return {
            "markdown": "# extracted title\n\nbody text",
            "source": str(path),
            "output_path": str(tmp_path / "out.md"),
            "duration_ms": 12,
            "chars": 22,
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

    def fake_convert(path, out_dir=None, backend=None, method=None, timeout_s=None):
        captured["path"] = str(path)
        captured["out_dir"] = out_dir
        return {
            "markdown": "## md",
            "source": str(path),
            "output_path": str(tmp_path / "x.md"),
            "duration_ms": 5,
            "chars": 5,
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
