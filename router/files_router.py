from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import settings

logger = logging.getLogger("router.files_router")

router = APIRouter(prefix="/v1")


def validate_upload(filename: str, content_type: str, size: int) -> str | None:
    allowed_lower = [t.lower() for t in settings.allowed_upload_mime_types]
    if content_type.lower() not in allowed_lower:
        return f"File type '{content_type}' is not allowed"

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size > max_bytes:
        return f"File exceeds maximum size of {settings.max_upload_size_mb} MB"

    if "/" in filename.lstrip("/") or ".." in filename:
        return "Invalid filename"

    return None


@router.post("/files")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "untitled"

    error = validate_upload(filename, content_type, len(content))
    if error:
        raise HTTPException(status_code=400, detail={"type": "invalid_request_error", "message": error})

    file_id = f"file_{uuid.uuid4().hex}"
    dest = settings.upload_dir / file_id
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    logger.info("Uploaded file_id=%s filename=%s type=%s size=%d", file_id, filename, content_type, len(content))

    return {
        "id": file_id,
        "filename": filename,
        "bytes": len(content),
        "created_at": dest.stat().st_ctime,
    }
