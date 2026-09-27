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


def test_openai_adapter_images_posts_to_images_endpoint():
    from providers.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(api_key="k", base_url="http://x/v1")
    with patch.object(adapter, "_post_json", return_value={"data": [{"b64_json": "Z"}]}) as post:
        out = adapter.images({"model": "m", "prompt": "p"})
    post.assert_called_once_with("/images/generations", {"model": "m", "prompt": "p"})
    assert out["data"] == [{"b64_json": "Z"}]


def test_gemini_adapter_images_normalizes_inline_data():
    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(api_key="k", base_url="http://g")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"inlineData": {"data": "B64"}}]}}]}
    with patch("providers.gemini_adapter.post_with_retry", return_value=mock_resp):
        out = adapter.images({"model": "gemini-2.5-flash-image", "prompt": "a cat", "n": 1})
    assert out["data"] == [{"b64_json": "B64"}]


class TestCloudImageDispatch:
    def _client(self) -> TestClient:
        from router.image_router import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_custom_provider_routes_to_cloud_adapter(self):
        cloud = MagicMock()
        cloud.images.return_value = {"created": 123, "data": [{"b64_json": "AAAA"}]}
        with patch(f"{IMAGE_ROUTER}._resolve_image_provider", return_value="agnes"), \
             patch(f"{IMAGE_ROUTER}.provider_router.adapter", return_value=cloud):
            client = self._client()
            resp = client.post("/v1/images/generations", json={"model": "agnes-image-2.1-flash", "prompt": "a cat"})
        assert resp.status_code == 200
        assert resp.json()["data"] == [{"b64_json": "AAAA"}]
        cloud.images.assert_called_once()
        assert cloud.images.call_args.args[0]["model"] == "agnes-image-2.1-flash"

    def test_cloud_provider_without_images_returns_501(self):
        plain = MagicMock(spec=[])
        with patch(f"{IMAGE_ROUTER}._resolve_image_provider", return_value="nvidia_nim"), \
             patch(f"{IMAGE_ROUTER}.provider_router.adapter", return_value=plain):
            client = self._client()
            resp = client.post("/v1/images/generations", json={"model": "some-image", "prompt": "a cat"})
        assert resp.status_code == 501

    def test_cloud_edits_return_501(self):
        with patch(f"{IMAGE_ROUTER}._resolve_image_provider", return_value="agnes"):
            client = self._client()
            resp = client.post("/v1/images/edits",
                data={"model": "agnes-image-2.1-flash", "prompt": "a cat", "n": 1},
                files={"image": ("t.png", b"x", "image/png")})
        assert resp.status_code == 501
