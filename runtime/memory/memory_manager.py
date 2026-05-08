from __future__ import annotations

import logging
from typing import Any

from runtime.memory.short_term import ShortTermMemory
from runtime.memory.semantic_memory import SemanticMemory, MemoryEntry
from runtime.memory.episodic_memory import EpisodicMemory, EpisodicRecord

logger = logging.getLogger("memory.manager")


class MemoryManager:
    def __init__(
        self,
        short_term: ShortTermMemory,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self.short_term = short_term
        self.semantic = semantic
        self.episodic = episodic

    async def record_execution(
        self,
        session_id: str = "",
        model: str = "",
        provider: str = "",
        task_summary: str = "",
        tool_calls: list[str] | None = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
        token_count: dict[str, int] | None = None,
    ) -> EpisodicRecord:
        record = self.episodic.record(
            session_id=session_id,
            model=model,
            provider=provider,
            task_summary=task_summary[:200] if task_summary else "",
            tool_calls=tool_calls or [],
            duration_ms=duration_ms,
            success=success,
            error=error,
            token_count=token_count,
        )

        if success and task_summary and session_id:
            self.semantic.store(
                text=task_summary,
                metadata={
                    "type": "execution",
                    "session_id": session_id,
                    "model": model,
                    "provider": provider,
                },
            )

        return record

    def search_relevant(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        results = self.semantic.search(query, top_k=top_k * 2)

        if session_id:
            results = [r for r in results if r.metadata.get("session_id") == session_id]

        return results[:top_k]

    def session_context(
        self, session_id: str, max_messages: int = 20,
    ) -> list[dict[str, Any]]:
        return self.short_term.get_context(session_id, max_messages=max_messages)

    def provider_reliability(self, provider: str) -> float:
        return self.episodic.success_rate(provider)

    def provider_avg_latency(self, provider: str) -> float:
        return self.episodic.avg_latency_ms(provider)
