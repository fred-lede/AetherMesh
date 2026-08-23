from __future__ import annotations

import json
import logging

import requests

from runtime.alerting.notifier_base import Alert, Notifier

logger = logging.getLogger("alerting.synology")


class SynologyChatNotifier(Notifier):
    name = "synology_chat"

    def __init__(self, webhook_url: str, timeout_s: int = 10) -> None:
        self.webhook_url = str(webhook_url).strip()
        self.timeout_s = timeout_s

    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            logger.warning("synology notifier missing webhook_url, skip")
            return False
        try:
            resp = requests.post(
                self.webhook_url,
                data={"payload": f'{{"text": {alert.format_text()!r}}}'.replace("'", '"')},
                timeout=self.timeout_s,
            )
            ok = resp.status_code == 200
            if not ok:
                logger.warning("synology chat send failed: %s %s", resp.status_code, resp.text[:200])
            return ok
        except requests.RequestException as exc:
            logger.warning("synology chat send error: %s", exc)
            return False
