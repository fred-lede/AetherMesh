from __future__ import annotations

from providers.ollama_adapter import _embedding_rows as ollama_embedding_rows
from providers.ollama_cloud_adapter import _embedding_rows as ollama_cloud_embedding_rows
from runtime.orchestration.capabilities import required_openai_capabilities


def test_ollama_embedding_rows_accept_plural_embeddings() -> None:
    rows = ollama_embedding_rows({"embeddings": [[0.1, 0.2, 0.3]]})

    assert rows == [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}]


def test_ollama_embedding_rows_accept_legacy_single_embedding() -> None:
    rows = ollama_embedding_rows({"embedding": [0.1] * 768})

    assert len(rows) == 1
    assert len(rows[0]["embedding"]) == 768


def test_ollama_embedding_rows_accept_openai_data_shape() -> None:
    rows = ollama_embedding_rows({
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        ],
    })

    assert rows == [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]


def test_ollama_cloud_embedding_rows_accept_legacy_single_embedding() -> None:
    rows = ollama_cloud_embedding_rows({"embedding": [0.1] * 768})

    assert len(rows) == 1
    assert len(rows[0]["embedding"]) == 768


def test_openai_embedding_payload_requires_embedding_capability() -> None:
    required = required_openai_capabilities({
        "model": "nomic-embed-text-v2-moe:latest",
        "input": "hello",
    })

    assert required == {"embeddings"}
