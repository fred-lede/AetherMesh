from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from runtime.documents.mineru_converter import MinerUError, convert_document, mineru_available

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("/health")
async def document_health() -> dict[str, Any]:
    return {
        "ok": True,
        "mineru_available": mineru_available(),
        "backend": "pipeline",
        "method": "auto",
    }


@router.post("/extract")
async def extract_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    tmp = Path(tempfile.gettempdir()) / f"aethermesh_doc_{uuid.uuid4().hex}{suffix}"
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file upload")
        tmp.write_bytes(data)
        result = await asyncio.to_thread(convert_document, tmp)
        return {"ok": True, "filename": file.filename or tmp.name, **result}
    except MinerUError as exc:
        return JSONResponse(status_code=422, content={"ok": False, "error": str(exc)})
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/extract/json")
async def extract_json(body: dict[str, Any]) -> dict[str, Any]:
    path = str(body.get("path") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path'")
    try:
        result = await asyncio.to_thread(
            convert_document,
            path,
            out_dir=str(body.get("output_dir") or "").strip() or None,
            backend=body.get("backend"),
            method=body.get("method"),
        )
        return {"ok": True, **result}
    except MinerUError as exc:
        return JSONResponse(status_code=422, content={"ok": False, "error": str(exc)})
