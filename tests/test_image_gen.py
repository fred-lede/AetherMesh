from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from providers.image_gen_adapter import ImageGenAdapter


def test_adapter_generate_returns_images():
    adapter = ImageGenAdapter()
    adapter.set_worker("http://test-host:11434")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "done": True}

    with patch("providers.image_gen_adapter.get_session") as mock_session_factory:
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session_factory.return_value = mock_session

        images = adapter.generate("x/z-image-turbo:fp8", "a cat", n=2)

    assert len(images) == 2
    assert images[0].startswith("iVBOR")
    assert images[1].startswith("iVBOR")
    assert mock_session.post.call_count == 2


def test_adapter_generate_handles_empty_image():
    adapter = ImageGenAdapter()
    adapter.set_worker("http://test-host:11434")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"image": "", "done": True}

    with patch("providers.image_gen_adapter.get_session") as mock_session_factory:
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session_factory.return_value = mock_session

        images = adapter.generate("x/z-image-turbo:fp8", "a cat", n=1)

    assert len(images) == 0


def test_adapter_set_worker():
    adapter = ImageGenAdapter()
    assert adapter.base_url == ""
    adapter.set_worker("http://192.168.1.100:11434")
    assert adapter.base_url == "http://192.168.1.100:11434"


IMAGE_ROUTER = "router.image_router"


@pytest.fixture
def mock_gen_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.generate.return_value = [
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    ]
    return adapter


@pytest.fixture
def gen_app(mock_gen_adapter: MagicMock) -> FastAPI:
    with patch(f"{IMAGE_ROUTER}.settings.image_gen_enabled", True):
        with patch(f"{IMAGE_ROUTER}.get_adapter", return_value=mock_gen_adapter):
            from router.image_router import router
            app = FastAPI()
            app.include_router(router)
            yield app


@pytest.fixture
def gen_client(gen_app: FastAPI) -> TestClient:
    return TestClient(gen_app)


class TestImageGenerations:
    def test_generations_returns_b64_json(self, gen_client: TestClient, mock_gen_adapter: MagicMock) -> None:
        resp = gen_client.post("/v1/images/generations", json={
            "model": "x/z-image-turbo:fp8",
            "prompt": "a cute cat",
            "n": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "created" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"].startswith("iVBOR")

    def test_generations_requires_prompt(self, gen_client: TestClient) -> None:
        resp = gen_client.post("/v1/images/generations", json={"model": "x/z-image-turbo:fp8"})
        assert resp.status_code == 422

    def test_generations_empty_prompt(self, gen_client: TestClient) -> None:
        resp = gen_client.post("/v1/images/generations", json={"model": "x/z-image-turbo:fp8", "prompt": ""})
        assert resp.status_code == 422


class TestImageEdits:
    def test_edits_aliases_to_generations(self, gen_client: TestClient) -> None:
        resp = gen_client.post("/v1/images/edits",
            data={
                "model": "x/z-image-turbo:fp8",
                "prompt": "a cute cat",
                "n": 1,
            },
            files={"image": ("test.png", b"fake", "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"].startswith("iVBOR")

    def test_edits_requires_prompt(self, gen_client: TestClient) -> None:
        resp = gen_client.post("/v1/images/edits",
            data={"prompt": ""},
            files={"image": ("test.png", b"fake", "image/png")},
        )
        assert resp.status_code == 400
