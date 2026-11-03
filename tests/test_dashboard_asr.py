from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.dashboard_server as dashboard_server


@pytest.mark.asyncio
async def test_asr_config_proxy_get_passthrough(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            return MagicMock(status_code=200, json=lambda: {"state": "ready", "version": 1}, text="")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("AIIH_API_KEY", "key-123")
    response = await dashboard_server._asr_config_proxy("GET")
    assert response.status_code == 200
    assert json.loads(response.body) == {"state": "ready", "version": 1}
    assert captured["url"] == f"{dashboard_server.ROUTER_BASE_URL}/api/asr/config"
    assert captured["headers"]["x-api-key"] == "key-123"


@pytest.mark.asyncio
async def test_asr_config_proxy_post_forwards_body_and_status(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content=None, headers=None):
            captured["url"] = url
            captured["content"] = content
            captured["headers"] = headers
            return MagicMock(status_code=400, json=lambda: {"detail": {"message": "bad"}}, text="")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    response = await dashboard_server._asr_config_proxy("POST", body=b'{"target_id": "x"}')
    assert response.status_code == 400
    assert json.loads(response.body) == {"detail": {"message": "bad"}}
    assert captured["content"] == b'{"target_id": "x"}'
    assert captured["headers"]["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_asr_config_proxy_router_down_maps_503(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            raise dashboard_server.httpx.ConnectError("refused")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    with pytest.raises(dashboard_server.HTTPException) as excinfo:
        await dashboard_server._asr_config_proxy("GET")
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_asr_config_proxy_non_json_response_passthrough_text(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return MagicMock(status_code=500, json=MagicMock(side_effect=ValueError("not json")), text="server exploded")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    response = await dashboard_server._asr_config_proxy("GET")
    assert response.status_code == 500
    assert json.loads(response.body) == {"message": "server exploded"}


def test_asr_config_get_requires_admin():
    app = FastAPI()
    app.include_router(dashboard_server.api)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/asr/config")
    assert resp.status_code in (401, 403)


def test_asr_config_post_requires_admin():
    app = FastAPI()
    app.include_router(dashboard_server.api)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/asr/config", json={"target_id": "local:cpu"})
    assert resp.status_code in (401, 403)
