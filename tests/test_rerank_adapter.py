from __future__ import annotations

from unittest.mock import MagicMock, patch

from providers.base import ProviderError
from providers.rerank_adapter import RerankAdapter
from runtime.orchestration.capabilities import required_openai_capabilities


def _fake_response(data: dict, ok: bool = True, status_code: int = 200, text: str = ""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = data
    return resp


def test_rerank_payload_requires_rerank_capability() -> None:
    required = required_openai_capabilities(
        {
            "model": "rerank/bge-reranker-v2-m3",
            "query": "capital of France",
            "documents": ["Paris is capital", "Apple is fruit"],
        }
    )
    assert required == {"rerank"}


def test_rerank_payload_not_confused_with_embeddings_or_chat() -> None:
    assert required_openai_capabilities({"input": "x"}) == {"embeddings"}
    assert required_openai_capabilities({"messages": []}) == {"chat"}


def test_rerank_converts_llamacpp_response_to_openai_shape() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    response = _fake_response(
        {
            "model": "bge",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
            "results": [
                {"index": 2, "relevance_score": -5.0752},
                {"index": 0, "relevance_score": 8.6535},
            ],
        }
    )
    with patch("providers.rerank_adapter.get_session") as mock_get:
        mock_get.return_value.post.return_value = response
        result = adapter.rerank(
            {
                "model": "rerank/bge-reranker-v2-m3",
                "query": "capital",
                "documents": ["Paris is capital", "Apple is fruit", "Eiffel tower"],
                "top_n": 3,
            }
        )

    assert result["object"] == "list"
    assert [r["index"] for r in result["data"]] == [0, 2]
    assert result["data"][0]["relevance_score"] == 8.6535
    assert result["data"][0]["document"] == "Paris is capital"
    assert result["data"][1]["document"] == "Eiffel tower"
    assert result["model"] == "rerank/bge-reranker-v2-m3"


def test_rerank_respects_top_n() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    response = _fake_response(
        {
            "results": [
                {"index": 0, "relevance_score": 5.0},
                {"index": 1, "relevance_score": 4.0},
            ]
        }
    )
    with patch("providers.rerank_adapter.get_session") as mock_get:
        mock_get.return_value.post.return_value = response
        result = adapter.rerank(
            {
                "query": "q",
                "documents": ["a", "b"],
                "top_n": 1,
            }
        )
    assert len(result["data"]) == 1
    assert result["data"][0]["index"] == 0


def test_rerank_requires_nonempty_documents() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    try:
        adapter.rerank({"query": "q", "documents": []})
    except ProviderError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("expected ProviderError")


def test_rerank_raises_on_server_error() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    response = _fake_response({}, ok=False, status_code=500, text="boom")
    with patch("providers.rerank_adapter.get_session") as mock_get:
        mock_get.return_value.post.return_value = response
        try:
            adapter.rerank({"query": "q", "documents": ["a"]})
        except ProviderError as exc:
            assert exc.status_code == 500
        else:
            raise AssertionError("expected ProviderError")


def test_rerank_unsupported_methods_raise() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    for method, payload in (
        (adapter.chat, {"messages": []}),
        (adapter.embeddings, {"input": "x"}),
        (adapter.stream, {}),
        (adapter.responses, {"input": "x"}),
    ):
        try:
            method(payload)
        except ProviderError:
            pass
        else:
            raise AssertionError(f"{method.__name__} should raise")


def test_rerank_health_check() -> None:
    adapter = RerankAdapter(base_url="http://127.0.0.1:11436")
    with patch("providers.rerank_adapter.get_session") as mock_get:
        mock_get.return_value.get.return_value = _fake_response({}, ok=True, status_code=200)
        result = adapter.health_check()
    assert result["ok"] is True
    assert result["base_url"] == "http://127.0.0.1:11436"