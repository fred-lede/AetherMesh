from __future__ import annotations

import json

from router.web_server_tools import (
    content_text,
    extract_fetch_url,
    extract_search_query,
    latest_user_text,
    stream_web_server_tool_response,
)


def _payloads(events: list[str]) -> list[dict]:
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data:"):
                payloads.append(json.loads(line.split(":", 1)[1].strip()))
    return payloads


def test_extracts_latest_user_text_from_blocks() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ignore"},
            {"role": "user", "content": [{"type": "text", "text": "特斯拉今日新聞"}]},
        ]
    }

    assert latest_user_text(payload) == "特斯拉今日新聞"
    assert content_text([{"type": "text", "text": "hello"}]) == "hello"


def test_query_and_url_extraction() -> None:
    assert extract_search_query("query: Tesla Q1 2026 earnings") == "Tesla Q1 2026 earnings"
    assert extract_search_query("特斯拉2026下半年財測") == "特斯拉2026下半年財測"
    assert extract_fetch_url("please fetch https://example.com/report?q=1.") == "https://example.com/report?q=1"


def test_streams_web_search_server_tool_response() -> None:
    def fake_search(query: str, max_results: int, timeout_s: int) -> list[dict[str, str]]:
        assert query == "特斯拉2026下半年財測"
        assert max_results == 2
        assert timeout_s == 3
        return [{"title": "Tesla outlook", "url": "https://example.com/tesla"}]

    events = list(
        stream_web_server_tool_response(
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "特斯拉2026下半年財測"}],
                "tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "tool", "name": "web_search"},
            },
            model="claude-sonnet-4-5",
            timeout_s=3,
            max_results=2,
            search_runner=fake_search,
        )
    )

    rendered = "".join(events)
    payloads = _payloads(events)
    assert '"type":"server_tool_use"' in rendered
    assert '"type":"web_search_tool_result"' in rendered
    assert "Tesla outlook" in rendered
    assert payloads[-1]["type"] == "message_stop"


def test_streams_web_fetch_server_tool_response() -> None:
    def fake_fetch(url: str, timeout_s: int) -> dict[str, str]:
        assert url == "https://example.com/report"
        assert timeout_s == 4
        return {
            "url": url,
            "title": "Report",
            "media_type": "text/html",
            "data": "Fetched report content",
        }

    events = list(
        stream_web_server_tool_response(
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "fetch https://example.com/report"}],
                "tools": [{"name": "web_fetch", "type": "web_fetch_20250305"}],
                "tool_choice": {"type": "tool", "name": "web_fetch"},
            },
            model="claude-sonnet-4-5",
            timeout_s=4,
            fetch_runner=fake_fetch,
        )
    )

    rendered = "".join(events)
    assert '"type":"web_fetch_tool_result"' in rendered
    assert "Fetched report content" in rendered
