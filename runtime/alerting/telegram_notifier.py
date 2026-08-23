from __future__ import annotations

import logging

import requests

from runtime.alerting.notifier_base import Alert, Notifier

logger = logging.getLogger("alerting.telegram")

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout_s: int = 10) -> None:
        self.bot_token = str(bot_token).strip()
        self.chat_id = str(chat_id).strip()
        self.timeout_s = timeout_s

    def send(self, alert: Alert) -> bool:
        if not self.bot_token or not self.chat_id:
            logger.warning("telegram notifier missing bot_token/chat_id, skip")
            return False
        try:
            resp = requests.post(
                _API_URL.format(token=self.bot_token),
                json={
                    "chat_id": self.chat_id,
                    "text": alert.format_text(),
                    "disable_web_page_preview": True,
                },
                timeout=self.timeout_s,
            )
            ok = resp.status_code == 200
            if not ok:
                logger.warning("telegram send failed: %s %s", resp.status_code, resp.text[:200])
            return ok
        except requests.RequestException as exc:
            logger.warning("telegram send error: %s", exc)
            return False
