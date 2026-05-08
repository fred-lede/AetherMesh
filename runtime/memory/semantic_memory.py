from __future__ import annotations

import logging
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("memory.semantic")


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    score: float = 0.0


class SemanticMemory:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._term_index: dict[str, dict[str, float]] = {}

    def store(
        self, text: str, metadata: dict[str, Any] | None = None,
    ) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry = MemoryEntry(
            id=entry_id,
            text=text,
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry
        terms = self._tokenize(text)
        total = sum(terms.values())
        self._term_index[entry_id] = {
            term: count / total for term, count in terms.items()
        }
        logger.debug("Stored semantic entry %s (%d terms)", entry_id, len(terms))
        return entry_id

    def search(
        self, query: str, top_k: int = 5,
    ) -> list[MemoryEntry]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        query_norm = math.sqrt(sum(v * v for v in query_terms.values()))

        scored: list[tuple[str, float]] = []
        for entry_id, term_index in self._term_index.items():
            dot = 0.0
            for term, qty in query_terms.items():
                if term in term_index:
                    dot += qty * term_index[term]

            doc_norm = math.sqrt(sum(v * v for v in term_index.values()))
            if query_norm == 0 or doc_norm == 0:
                similarity = 0.0
            else:
                similarity = dot / (query_norm * doc_norm)

            if similarity > 0:
                scored.append((entry_id, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results: list[MemoryEntry] = []
        for entry_id, similarity in top:
            entry = self._entries.get(entry_id)
            if entry:
                entry.score = round(similarity, 4)
                results.append(entry)

        return results

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)
        self._term_index.pop(entry_id, None)

    def clear(self) -> None:
        self._entries.clear()
        self._term_index.clear()

    def count(self) -> int:
        return len(self._entries)

    @staticmethod
    def _tokenize(text: str) -> Counter:
        tokens: list[str] = []
        word: list[str] = []
        for ch in text.lower():
            if ch.isalnum() or ch in ("_", "-"):
                word.append(ch)
            else:
                if word:
                    t = "".join(word)
                    if len(t) > 1:
                        tokens.append(t)
                    word = []
        if word:
            t = "".join(word)
            if len(t) > 1:
                tokens.append(t)
        return Counter(tokens)
