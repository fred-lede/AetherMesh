from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.alerting.notifier_base import Alert, Notifier, Severity
from runtime.alerting.synology_notifier import SynologyChatNotifier
from runtime.alerting.telegram_notifier import TelegramNotifier

logger = logging.getLogger("alerting.manager")

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _severity_rank(severity: Severity) -> int:
    return _SEVERITY_RANK[severity.value]


@dataclass(slots=True)
class RuleState:
    last_fired: float = 0.0


class AlertManager:
    def __init__(self, config_path: str | Path | None = None, cooldown_s: int = 300) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._rule_states: dict[str, RuleState] = {}
        self._config: dict[str, Any] = {}
        self._last_mtime: float = -1.0
        if self.config_path is not None:
            self.load_config(self.config_path)

    def load_config(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            self._config = {}
            return
        try:
            self._last_mtime = p.stat().st_mtime
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("failed to load notification config: %s", p)
            self._config = {}
            return
        self._config = data if isinstance(data, dict) else {}

    def reload_if_changed(self) -> bool:
        if self.config_path is None:
            return False
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._last_mtime:
            return False
        self.load_config(self.config_path)
        return True

    def save_config(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.config_path
        if target is None:
            return
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._config, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def update_config(self, data: dict[str, Any]) -> None:
        channels = data.get("channels", {})
        if not isinstance(channels, dict):
            return
        current = self._config.setdefault("channels", {})
        secret_keys = {"bot_token", "webhook_url"}
        if not isinstance(current, dict):
            current = {}
            self._config["channels"] = current
        for name, updates in channels.items():
            if not isinstance(updates, dict):
                continue
            merged = current.get(name)
            if not isinstance(merged, dict):
                merged = {}
            for key, value in updates.items():
                if key in secret_keys and not str(value or "").strip():
                    continue
                merged[key] = value
            current[name] = merged

    def _channel_cfg(self, name: str) -> dict[str, Any]:
        channels = self._config.get("channels", {})
        cfg = channels.get(name, {}) if isinstance(channels, dict) else {}
        return cfg if isinstance(cfg, dict) else {}

    def _build_notifiers(self) -> list[tuple[Notifier, Severity]]:
        notifiers: list[tuple[Notifier, Severity]] = []

        tg = self._channel_cfg("telegram")
        if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
            min_sev = Severity(str(tg.get("min_severity", "warning")))
            notifiers.append((TelegramNotifier(tg["bot_token"], tg["chat_id"]), min_sev))

        sc = self._channel_cfg("synology_chat")
        if sc.get("enabled") and sc.get("webhook_url"):
            min_sev = Severity(str(sc.get("min_severity", "warning")))
            notifiers.append((SynologyChatNotifier(sc["webhook_url"]), min_sev))

        return notifiers

    def dispatch(
        self,
        title: str,
        message: str,
        severity: Severity = Severity.WARNING,
        rule_key: str = "",
        cooldown_s: int | None = None,
    ) -> int:
        cooldown = self.cooldown_s if cooldown_s is None else cooldown_s
        now = time.time()
        with self._lock:
            if rule_key:
                state = self._rule_states.setdefault(rule_key, RuleState())
                if now - state.last_fired < cooldown:
                    logger.debug("alert %s suppressed by cooldown", rule_key)
                    return 0
                state.last_fired = now

        alert = Alert(title=title, message=message, severity=severity)
        sent = 0
        for notifier, min_sev in self._build_notifiers():
            if _severity_rank(severity) < _severity_rank(min_sev):
                continue
            if notifier.send(alert):
                sent += 1
        if sent == 0:
            logger.warning("alert %s not delivered to any channel", rule_key or title)
        return sent

    def send_test(self, channel: str) -> tuple[bool, str]:
        test_alert = Alert(
            title="AetherMesh 測試通知",
            message="如果你看到這則訊息，表示此通道設定正確。",
            severity=Severity.INFO,
            source="dashboard-test",
        )
        if channel == "telegram":
            cfg = self._channel_cfg("telegram")
            if not (cfg.get("bot_token") and cfg.get("chat_id")):
                return False, "Telegram 未設定 bot_token / chat_id"
            ok = TelegramNotifier(cfg["bot_token"], cfg["chat_id"]).send(test_alert)
        elif channel == "synology_chat":
            cfg = self._channel_cfg("synology_chat")
            if not cfg.get("webhook_url"):
                return False, "Synology Chat 未設定 webhook URL"
            ok = SynologyChatNotifier(cfg["webhook_url"]).send(test_alert)
        else:
            return False, f"unknown channel: {channel}"
        return ok, ("sent" if ok else "send failed, check token/url/log")


_shared: AlertManager | None = None
_shared_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                from config.settings import settings

                _shared = AlertManager(config_path=settings.config_path("notifications.json"))
    return _shared
