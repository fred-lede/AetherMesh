from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from runtime.security.tool_policy import forced_server_tool


SERVER_TOOL_USE = "server_tool_use"
WEB_SEARCH_TOOL_RESULT = "web_search_tool_result"
WEB_FETCH_TOOL_RESULT = "web_fetch_tool_result"

SearchRunner = Callable[[str, int, int], list[dict[str, str]]]
FetchRunner = Callable[[str, int], dict[str, str]]


def format_anthropic_sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=True, separators=(',', ':'))}\n\n"


def latest_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return content_text(message.get("content", ""))
    return ""


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    parts.append(content_text(block.get("content", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _strip_system_tags(text: str) -> str:
    return re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.DOTALL).strip()


def extract_search_query(text: str) -> str:
    clean = " ".join(str(text or "").split())
    clean = _strip_system_tags(clean)
    for marker in ("query:", "search:", "搜尋:", "查詢:"):
        index = clean.lower().find(marker.lower())
        if index != -1:
            value = clean[index + len(marker):].strip()
            if value:
                return value
    return clean


def extract_fetch_url(text: str) -> str:
    match = re.search(r"https?://[^\s<>'\"]+", str(text or ""))
    return match.group(0).rstrip(".,);]") if match else ""


def run_web_search(query: str, max_results: int, timeout_s: int) -> list[dict[str, str]]:
    if not query:
        return []
    from runtime.tools.web_search import web_search_manager
    results = web_search_manager.search(query, max_results=max_results)
    return [{"title": r.title, "url": r.url} for r in results]


def run_web_fetch(url: str, timeout_s: int) -> dict[str, str]:
    if not url:
        raise ValueError("web_fetch requires a URL in the latest user message")
    from runtime.tools.web_search import web_search_manager
    raw = web_search_manager.fetch_url(url, timeout_s=timeout_s)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else url
    return {
        "url": url,
        "title": title,
        "media_type": "text/html",
        "data": _strip_html(raw)[:20_000],
    }


def stream_web_server_tool_response(
    payload: dict[str, Any],
    *,
    model: str,
    input_tokens: int = 0,
    timeout_s: int = 15,
    max_results: int = 5,
    search_runner: SearchRunner = run_web_search,
    fetch_runner: FetchRunner = run_web_fetch,
) -> Iterable[str]:
    tool_name = forced_server_tool(payload)
    if tool_name not in {"web_search", "web_fetch"}:
        return

    text = latest_user_text(payload)
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = time.time()
    tool_input = {"query": extract_search_query(text)} if tool_name == "web_search" else {"url": extract_fetch_url(text)}

    yield format_anthropic_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )
    yield format_anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": SERVER_TOOL_USE, "id": tool_id, "name": tool_name, "input": tool_input},
        },
    )
    yield format_anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    try:
        if tool_name == "web_search":
            results = search_runner(str(tool_input["query"]), max_results, timeout_s)
            result_content: Any = [
                {"type": "web_search_result", "title": result["title"], "url": result["url"]}
                for result in results
            ]
            summary = _search_summary(str(tool_input["query"]), results)
            result_block_type = WEB_SEARCH_TOOL_RESULT
        else:
            fetched = fetch_runner(str(tool_input["url"]), timeout_s)
            result_content = {
                "type": "web_fetch_result",
                "url": fetched["url"],
                "content": {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": fetched["media_type"],
                        "data": fetched["data"],
                    },
                    "title": fetched["title"],
                    "citations": {"enabled": True},
                },
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            summary = fetched["data"][:4000]
            result_block_type = WEB_FETCH_TOOL_RESULT
    except Exception as exc:
        result_block_type = WEB_SEARCH_TOOL_RESULT if tool_name == "web_search" else WEB_FETCH_TOOL_RESULT
        result_content = {"type": f"{tool_name}_error", "error_code": "unavailable"}
        summary = f"{tool_name} unavailable: {type(exc).__name__}"

    yield format_anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": result_block_type,
                "tool_use_id": tool_id,
                "content": result_content,
            },
        },
    )
    yield format_anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 1})
    yield format_anthropic_sse(
        "content_block_start",
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
    )
    yield format_anthropic_sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": summary}},
    )
    yield format_anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 2})
    output_tokens = max(1, len(summary) // 4)
    usage_key = "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    yield format_anthropic_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "server_tool_use": {usage_key: 1},
                "latency_ms": int((time.time() - started) * 1000),
            },
        },
    )
    yield format_anthropic_sse("message_stop", {"type": "message_stop"})


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}")
    summary = "\n\n".join(lines)
    references = "\n\n---\nReferences:"
    for index, result in enumerate(results, start=1):
        references += f"\n{index}. {result['url']}"
    return summary + references


def _strip_html(value: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return " ".join(html.unescape(without_tags).split())
