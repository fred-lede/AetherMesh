from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from runtime.documents.job_manager import DocumentJobManager
from runtime.documents.mineru_converter import MinerUError, convert_document, mineru_available

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_job_manager = DocumentJobManager()


@router.get("/health")
async def document_health() -> dict[str, Any]:
    return {
        "ok": True,
        "mineru_available": mineru_available(),
        "backend": "pipeline",
        "method": "auto",
    }


@router.post("/extract")
async def extract_upload(
    file: UploadFile = File(...),
    include_images: bool = Query(False, description="Embed images as base64 data URIs in markdown"),
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    tmp = Path(tempfile.gettempdir()) / f"aethermesh_doc_{uuid.uuid4().hex}{suffix}"
    out_dir = Path(tempfile.mkdtemp(prefix="aethermesh_mineru_"))
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file upload")
        tmp.write_bytes(data)
        result = await asyncio.to_thread(
            convert_document, tmp, out_dir=out_dir, include_images=include_images
        )
        return {"ok": True, "filename": file.filename or tmp.name, **result}
    except MinerUError as exc:
        return JSONResponse(status_code=422, content={"ok": False, "error": str(exc)})
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)
        except OSError:
            pass


@router.post("/extract/async")
async def extract_async(
    file: UploadFile = File(...),
    include_images: bool = Query(False, description="Embed images as base64 data URIs in markdown"),
) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    job_id = _job_manager.submit(
        data,
        file.filename or "upload.pdf",
        include_images=include_images,
    )
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = _job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, **job}


@router.get("/jobs")
async def job_list(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return {"ok": True, "jobs": _job_manager.list(limit=limit)}


@router.post("/extract/json")
async def extract_json(
    body: dict[str, Any],
    include_images: bool = Query(False, description="Embed images as base64 data URIs in markdown"),
) -> dict[str, Any]:
    path = str(body.get("path") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path'")
    user_out_dir = str(body.get("output_dir") or "").strip()
    out_dir: str | None = user_out_dir or tempfile.mkdtemp(prefix="aethermesh_mineru_")
    try:
        result = await asyncio.to_thread(
            convert_document,
            path,
            out_dir=out_dir,
            backend=body.get("backend"),
            method=body.get("method"),
            include_images=include_images,
        )
        return {"ok": True, **result}
    except MinerUError as exc:
        return JSONResponse(status_code=422, content={"ok": False, "error": str(exc)})
    finally:
        if not user_out_dir:
            try:
                import shutil

                shutil.rmtree(out_dir, ignore_errors=True)
            except OSError:
                pass


@router.post("/extract/json/async")
async def extract_json_async(body: dict[str, Any]) -> dict[str, Any]:
    path = str(body.get("path") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path'")
    data = Path(path)
    if not data.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    job_id = _job_manager.submit(
        data.read_bytes(),
        data.name,
        include_images=bool(body.get("include_images", False)),
        backend=body.get("backend"),
        method=body.get("method"),
        timeout_s=body.get("timeout_s"),
    )
    return {"ok": True, "job_id": job_id, "status": "queued"}
