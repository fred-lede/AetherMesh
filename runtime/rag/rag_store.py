from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger("runtime.rag.rag_store")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    blocks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > chunk_size:
            blocks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        blocks.append(current)
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
            continue
        start = 0
        while start < len(block):
            end = min(start + chunk_size, len(block))
            chunk = block[start:end]
            if chunks and overlap > 0:
                chunks.append(f"{chunks[-1][-overlap:]} {chunk}")
            else:
                chunks.append(chunk)
            if end >= len(block):
                break
            start = end - overlap
    return chunks


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keyword_score(query_tokens: list[str], text: str) -> float:
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        score += counts.get(token, 0)
    return score / math.sqrt(len(text_tokens) + 1)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else Path("config") / "rag_store.json"
        self._lock = threading.RLock()
        self._chunks: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable RAG store %s", self._path)
            return
        if isinstance(data, dict):
            self._chunks = data.get("chunks") or {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"chunks": self._chunks}, ensure_ascii=False),
            encoding="utf-8",
        )

    def ingest(self, text: str, metadata: dict[str, Any] | None = None, embedding: list[float] | None = None) -> list[str]:
        ids: list[str] = []
        with self._lock:
            for chunk in chunk_text(text):
                chunk_id = f"chunk_{uuid.uuid4().hex[:16]}"
                self._chunks[chunk_id] = {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata or {},
                    "embedding": embedding,
                    "created_at": time.time(),
                }
                ids.append(chunk_id)
            self._save()
        return ids

    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            removed = self._chunks.pop(chunk_id, None) is not None
            if removed:
                self._save()
            return removed

    def clear(self) -> int:
        with self._lock:
            count = len(self._chunks)
            self._chunks.clear()
            self._save()
            return count

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedding: list[float] | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            chunks = list(self._chunks.values())
        if not chunks:
            return []
        if filter_metadata:
            chunks = [
                c
                for c in chunks
                if all(c.get("metadata", {}).get(k) == v for k, v in filter_metadata.items())
            ]
        query_tokens = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in chunks:
            if embedding is not None:
                score = _cosine(embedding, chunk.get("embedding") or [])
            else:
                score = _keyword_score(query_tokens, chunk.get("text", ""))
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, chunk in scored[: max(1, top_k)]:
            entry = dict(chunk)
            entry["score"] = round(score, 4)
            results.append(entry)
        return results


rag_store = RagStore(settings.config_path("rag_store.json"))
