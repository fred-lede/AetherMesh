from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from config.settings import settings
from runtime.rag.rag_store import rag_store
from runtime.tools.file_parser import parse_file

logger = logging.getLogger("router.rag_router")


def _compute_embeddings(service, texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return None
    try:
        result = service.handle_embeddings(
            {"input": texts, "model": settings.rag_embedding_model}
        )
    except Exception:
        logger.warning("RAG embedding computation failed; falling back to keyword search", exc_info=True)
        return None
    data = result.get("data") or []
    embeddings = [d.get("embedding") for d in data if isinstance(d, dict)]
    if embeddings and all(isinstance(e, list) and e for e in embeddings):
        return embeddings
    return None


def create_rag_router(service: Any):
    router = APIRouter(prefix="/v1/rag")

    @router.post("/ingest")
    def ingest(payload: dict[str, Any] = Body(...)):
        text = payload.get("text")
        file_id = payload.get("file_id")
        if not text and not file_id:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'file_id'")
        if file_id:
            source = Path(settings.upload_dir) / file_id
            if not source.exists():
                raise HTTPException(status_code=400, detail=f"File '{file_id}' not found")
            try:
                mime_type = mimetypes.guess_type(file_id)[0] or "application/octet-stream"
                parsed = parse_file(source, mime_type, filename=file_id)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc
            text = (text or "") + "\n" + parsed.text
        if not text or not str(text).strip():
            raise HTTPException(status_code=400, detail="Ingest text is empty")
        metadata = payload.get("metadata") or {}
        if file_id:
            metadata.setdefault("file_id", file_id)
        embeddings = None
        if payload.get("embed", False) or payload.get("embedding") is not None:
            embeddings = _compute_embeddings(service, rag_store.chunk_text(str(text)))
        ids = rag_store.ingest(str(text), metadata=metadata, embedding=payload.get("embedding"))
        return {"object": "rag_ingest", "chunk_count": len(ids), "chunk_ids": ids}

    @router.post("/search")
    def search(payload: dict[str, Any] = Body(...)):
        query = payload.get("query")
        if not query or not str(query).strip():
            raise HTTPException(status_code=400, detail="Missing required field 'query'")
        embedding = None
        if payload.get("embed", False):
            embeddings = _compute_embeddings(service, [str(query)])
            if embeddings:
                embedding = embeddings[0]
        results = rag_store.search(
            str(query),
            top_k=int(payload.get("top_k", 5)),
            embedding=embedding,
            filter_metadata=payload.get("filter"),
        )
        return {"object": "rag_search", "query": query, "data": results}

    @router.get("/stats")
    def stats():
        return {"object": "rag_stats", "chunk_count": rag_store.count(), "path": str(rag_store._path)}

    @router.delete("/chunks/{chunk_id}")
    def delete_chunk(chunk_id: str):
        if not rag_store.delete(chunk_id):
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
        return {"deleted": True}

    @router.delete("")
    def clear_all():
        removed = rag_store.clear()
        return {"deleted": removed}

    return router
