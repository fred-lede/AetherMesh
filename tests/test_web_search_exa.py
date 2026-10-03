from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from runtime.tools.web_search.exa import ExaSearchProvider
from runtime.tools.web_search.search_provider import SearchProviderError


def test_exa_not_configured_raises(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider = ExaSearchProvider(api_key="")
    assert provider.configured is False
    with pytest.raises(SearchProviderError):
        provider.search("q")


def test_exa_parses_results():
    provider = ExaSearchProvider(api_key="k")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"title": "T1", "url": "https://a", "text": "hello world", "publishedDate": "2024-01-01"},
        ]
    }
    with patch("requests.post", return_value=mock_resp) as post:
        results = provider.search("q", max_results=3)
    assert results[0].title == "T1"
    assert results[0].url == "https://a"
    assert results[0].content == "hello world"
    body = post.call_args.kwargs["json"]
    assert body["query"] == "q"
    assert body["numResults"] == 3
    assert post.call_args.kwargs["headers"]["x-api-key"] == "k"


def test_exa_maps_http_error_status():
    provider = ExaSearchProvider(api_key="k")
    err = requests.HTTPError("429")
    err.response = MagicMock(status_code=429)
    with patch("requests.post", side_effect=err):
        with pytest.raises(SearchProviderError) as exc:
            provider.search("q")
    assert exc.value.status_code == 429


def test_manager_prioritizes_exa():
    from runtime.tools.web_search import WebSearchManager

    assert WebSearchManager().providers[0].name == "exa"
