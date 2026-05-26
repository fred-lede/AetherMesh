from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from router.files_router import validate_upload


@pytest.fixture
def mock_settings():
    with patch("router.files_router.settings") as mock:
        mock.upload_dir = Path(tempfile.mkdtemp())
        mock.max_upload_size_mb = 1
        mock.allowed_upload_mime_types = ["text/plain", "application/pdf"]
        yield mock
        for f in mock.upload_dir.iterdir():
            f.unlink()
        mock.upload_dir.rmdir()


def test_validate_upload_accepts_allowed_type(mock_settings):
    result = validate_upload("hello.txt", "text/plain", 100)
    assert result is None


def test_validate_upload_rejects_disallowed_type(mock_settings):
    result = validate_upload("script.exe", "application/x-msdownload", 100)
    assert result is not None
    assert "type" in result


def test_validate_upload_rejects_oversized(mock_settings):
    result = validate_upload("big.txt", "text/plain", mock_settings.max_upload_size_mb * 1024 * 1024 + 1)
    assert result is not None
    assert "size" in result


def test_validate_upload_rejects_path_traversal(mock_settings):
    result = validate_upload("../../etc/passwd", "text/plain", 100)
    assert result is not None


@pytest.mark.asyncio
async def test_upload_endpoint_success(mock_settings):
    content = b"Hello, this is a test file."
    headers = Headers({"content-type": "text/plain"})
    upload = UploadFile(file=io.BytesIO(content), filename="test.txt", headers=headers)
    from router.files_router import upload_file as upload_handler
    response = await upload_handler(upload)
    assert response["filename"] == "test.txt"
    assert response["bytes"] == len(content)
    assert response["id"].startswith("file_")
    assert (mock_settings.upload_dir / response["id"]).exists()


@pytest.mark.asyncio
async def test_upload_endpoint_rejects_bad_type(mock_settings):
    upload = UploadFile(filename="evil.exe", file=io.BytesIO(b"bad"))
    from router.files_router import upload_file as upload_handler
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await upload_handler(upload)
