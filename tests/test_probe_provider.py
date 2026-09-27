from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import HTTPException

from dashboard.dashboard_server import _probe_provider


def test_probe_rerank_healthy():
    resp = MagicMock(ok=True)
    resp.elapsed.total_seconds.return_value = 0.05
    with patch("dashboard.dashboard_server.get_session") as factory:
        factory.return_value.get.return_value = resp
        result = _probe_provider("rerank")
    assert result["ok"] is True
    assert result["status"] == "healthy"
    assert result["base_url"]


def test_probe_rerank_unreachable():
    with patch("dashboard.dashboard_server.get_session") as factory:
        factory.return_value.get.side_effect = requests.RequestException("nope")
        result = _probe_provider("rerank")
    assert result["ok"] is False
    assert result["status"] == "unreachable"


def test_probe_local_aux_returns_message():
    result = _probe_provider("image_gen")
    assert result["ok"] is False
    assert result["status"] == "no_http_probe"


def test_probe_unknown_still_404():
    with pytest.raises(HTTPException) as exc:
        _probe_provider("bogus")
    assert exc.value.status_code == 404
