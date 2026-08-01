from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from router.streaming_router import async_stream_response, format_sse_event, stream_response


def test_format_sse_event_dict() -> None:
    assert format_sse_event({"type": "ok"}) == "data: {\"type\": \"ok\"}\n\n"


def test_format_sse_event_string() -> None:
    assert format_sse_event("done") == "data: done\n\n"


@pytest.mark.asyncio
async def test_stream_response_yields_all_items() -> None:
    response = stream_response([{"a": 1}, "done"])
    chunks = [chunk async for chunk in response.body_iterator]
    assert "".join(chunks) == "data: {\"a\": 1}\n\ndata: done\n\n"


@pytest.mark.asyncio
async def test_async_stream_response_exhausts_cleanly() -> None:
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)

    response = await async_stream_response([{"a": 1}, "done"], request)
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == ["data: {\"a\": 1}\n\n", "data: done\n\n"]


@pytest.mark.asyncio
async def test_async_stream_response_empty_iterator() -> None:
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)

    response = await async_stream_response([], request)
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == []
