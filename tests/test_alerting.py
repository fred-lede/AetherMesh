from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from runtime.alerting.alert_manager import AlertManager
from runtime.alerting.notifier_base import Alert, Severity
from runtime.alerting.synology_notifier import SynologyChatNotifier
from runtime.alerting.telegram_notifier import TelegramNotifier


def _alert(severity: Severity = Severity.WARNING) -> Alert:
    return Alert(title="t", message="m", severity=severity)


class TestTelegramNotifier:
    def test_send_success(self, monkeypatch) -> None:
        resp = MagicMock(status_code=200)
        posted = {}

        def fake_post(url, **kwargs):
            posted["url"] = url
            posted["json"] = kwargs.get("json")
            return resp

        monkeypatch.setattr("runtime.alerting.telegram_notifier.requests.post", fake_post)
        n = TelegramNotifier("tok123", "chat42")
        assert n.send(_alert()) is True
        assert "bottok123/sendMessage" in posted["url"]
        body = posted["json"]
        assert body["chat_id"] == "chat42"
        assert "[WARNING]" in body["text"]

    def test_send_non_200_returns_false(self, monkeypatch) -> None:
        resp = MagicMock(status_code=401)
        monkeypatch.setattr(
            "runtime.alerting.telegram_notifier.requests.post", lambda url, **kw: resp
        )
        n = TelegramNotifier("bad", "c")
        assert n.send(_alert()) is False

    def test_missing_token_skips_http(self, monkeypatch) -> None:
        called = []

        def fail_post(*a, **kw):
            called.append(1)
            raise AssertionError("should not call http")

        monkeypatch.setattr("runtime.alerting.telegram_notifier.requests.post", fail_post)
        n = TelegramNotifier("", "")
        assert n.send(_alert()) is False
        assert called == []

    def test_network_error_returns_false(self, monkeypatch) -> None:
        import requests as requests_lib

        def boom(url, **kw):
            raise requests_lib.ConnectionError("down")

        monkeypatch.setattr("runtime.alerting.telegram_notifier.requests.post", boom)
        n = TelegramNotifier("tok", "c")
        assert n.send(_alert()) is False


class TestSynologyNotifier:
    def test_send_success_payload_json(self, monkeypatch) -> None:
        import requests as requests_lib

        captured = {}

        def fake_post(url, data=None, timeout=None):
            captured["data"] = data
            return MagicMock(status_code=200)

        monkeypatch.setattr("runtime.alerting.synology_notifier.requests.post", fake_post)
        n = SynologyChatNotifier("https://nas/webhook")
        assert n.send(_alert(Severity.CRITICAL)) is True
        payload = json.loads(captured["data"]["payload"])
        assert payload["text"].startswith("[CRITICAL]")

    def test_missing_webhook_skips(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "runtime.alerting.synology_notifier.requests.post",
            lambda *a, **kw: pytest.fail("no http"),
        )
        assert SynologyChatNotifier("").send(_alert()) is False


class TestAlertManager:
    def _manager(self, tmp_path, channels=None):
        cfg = tmp_path / "notifications.json"
        mgr = AlertManager(config_path=cfg)
        mgr.config["channels"] = channels or {}
        return mgr, cfg

    def test_dispatch_sends_to_enabled_channel(self, tmp_path) -> None:
        mgr, _ = self._manager(
            tmp_path,
            {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "c"}},
        )
        with patch.object(TelegramNotifier, "send", return_value=True) as mock_send:
            sent = mgr.dispatch("title", "msg", Severity.WARNING, rule_key="r1")
        assert sent == 1
        mock_send.assert_called_once()

    def test_cooldown_suppresses_second_alert(self, tmp_path) -> None:
        mgr, _ = self._manager(
            tmp_path,
            {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "c"}},
        )
        with patch.object(TelegramNotifier, "send", return_value=True) as mock_send:
            assert mgr.dispatch("a", "m", rule_key="r", cooldown_s=600) == 1
            assert mgr.dispatch("b", "m", rule_key="r", cooldown_s=600) == 0
        assert mock_send.call_count == 1

    def test_min_severity_filters(self, tmp_path) -> None:
        mgr, _ = self._manager(
            tmp_path,
            {
                "telegram": {
                    "enabled": True,
                    "bot_token": "tok",
                    "chat_id": "c",
                    "min_severity": "critical",
                }
            },
        )
        with patch.object(TelegramNotifier, "send", return_value=True) as mock_send:
            assert mgr.dispatch("a", "m", severity=Severity.WARNING) == 0
            assert mgr.dispatch("a", "m", severity=Severity.CRITICAL) == 1
        assert mock_send.call_count == 1

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        mgr, cfg_path = self._manager(tmp_path)
        mgr.config["channels"] = {"telegram": {"enabled": True, "bot_token": "s3cret"}}
        mgr.save_config()
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert loaded["channels"]["telegram"]["bot_token"] == "s3cret"
        mgr2 = AlertManager(config_path=cfg_path)
        assert mgr2.config["channels"]["telegram"]["enabled"] is True

    def test_update_config_keeps_empty_secret(self, tmp_path) -> None:
        mgr, _ = self._manager(tmp_path)
        mgr.config["channels"] = {
            "telegram": {"enabled": False, "bot_token": "keep-me", "chat_id": ""}
        }
        mgr.update_config({"channels": {"telegram": {"enabled": True, "bot_token": ""}}})
        ch = mgr.config["channels"]["telegram"]
        assert ch["enabled"] is True
        assert ch["bot_token"] == "keep-me"

    def test_reload_if_changed_picks_up_new_mtime(self, tmp_path) -> None:
        cfg_path = tmp_path / "notifications.json"
        cfg_path.write_text(json.dumps({"channels": {}}), encoding="utf-8")
        mgr = AlertManager(config_path=cfg_path)
        assert mgr.reload_if_changed() is False
        import os
        import time as time_mod

        future = time_mod.time() + 10
        os.utime(cfg_path, (future, future))
        cfg_path.write_text(
            json.dumps({"watchdog": {"interval_s": 99}}), encoding="utf-8"
        )
        assert mgr.reload_if_changed() is True
        assert mgr.config["watchdog"]["interval_s"] == 99

    def test_send_test_unknown_channel(self, tmp_path) -> None:
        mgr, _ = self._manager(tmp_path)
        ok, msg = mgr.send_test("carrier_pigeon")
        assert ok is False
        assert "unknown channel" in msg

    def test_send_test_telegram_requires_config(self, tmp_path) -> None:
        mgr, _ = self._manager(tmp_path)
        ok, msg = mgr.send_test("telegram")
        assert ok is False
