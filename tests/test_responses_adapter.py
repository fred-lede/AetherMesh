from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import MagicMock

from router.openai.responses_adapter import _stream_with_keepalive


def test_keepalive_stream_aborts_adapter_when_client_closes():
    released = Event()
    adapter = MagicMock()
    adapter.abort_stream.side_effect = released.set

    def blocked_stream():
        yield "data: {\"type\": \"response.created\"}\\n\\n"
        released.wait(timeout=2)

    async def consume_then_close() -> None:
        stream = _stream_with_keepalive(blocked_stream(), adapter=adapter)
        await anext(stream)
        await stream.aclose()

    asyncio.run(consume_then_close())

    adapter.abort_stream.assert_called_once_with()
