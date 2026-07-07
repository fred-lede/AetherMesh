from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.image_gen_adapter import ImageGenAdapter


@pytest.mark.asyncio
async def test_adapter_generate_returns_images():
    adapter = ImageGenAdapter()
    adapter.set_worker("http://test-host:11434")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "done": True}

    with patch("providers.image_gen_adapter.get_session") as mock_session_factory:
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_resp
        mock_session_factory.return_value = mock_session

        images = await adapter.generate("x/z-image-turbo:fp8", "a cat", n=2)

    assert len(images) == 2
    assert images[0].startswith("iVBOR")
    assert images[1].startswith("iVBOR")
    assert mock_session.post.call_count == 2


@pytest.mark.asyncio
async def test_adapter_generate_handles_empty_image():
    adapter = ImageGenAdapter()
    adapter.set_worker("http://test-host:11434")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"image": "", "done": True}

    with patch("providers.image_gen_adapter.get_session") as mock_session_factory:
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_resp
        mock_session_factory.return_value = mock_session

        images = await adapter.generate("x/z-image-turbo:fp8", "a cat", n=1)

    assert len(images) == 0


@pytest.mark.asyncio
async def test_adapter_set_worker():
    adapter = ImageGenAdapter()
    assert adapter.base_url == ""
    adapter.set_worker("http://192.168.1.100:11434")
    assert adapter.base_url == "http://192.168.1.100:11434"
