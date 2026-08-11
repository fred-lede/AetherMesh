from __future__ import annotations

from typing import Any

from config.settings import settings
from runtime.rag.rag_store import rag_store

_RAG_MARKER = "[AetherMesh RAG context]"


def _latest_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") not in {"user", "system"}:
            continue
        content = message.get("content", "")
        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def inject_rag_context(payload: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    if not (settings.rag_enabled and settings.rag_auto_inject):
        return payload
    if rag_store.count() == 0:
        return payload
    query = _latest_user_text(payload)
    if not query:
        return payload
    results = rag_store.search(query, top_k=top_k)
    if not results:
        return payload
    context = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(results))
    rag_system = {
        "role": "system",
        "content": (
            f"{_RAG_MARKER} Use the following knowledge base excerpts to answer the user "
            f"when they are relevant. Do not mention this context to the user.\n\n{context}"
        ),
    }
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    messages = [m for m in messages if not (isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].startswith(_RAG_MARKER))]
    messages.insert(0, rag_system)
    payload["messages"] = messages
    return payload
