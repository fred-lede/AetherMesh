from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("alerting.notifier")


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_PREFIX = {
    Severity.INFO: "[INFO]",
    Severity.WARNING: "[WARNING]",
    Severity.CRITICAL: "[CRITICAL]",
}


@dataclass(slots=True)
class Alert:
    title: str
    message: str
    severity: Severity = Severity.WARNING
    source: str = "watchdog"
    timestamp: float = field(default_factory=time.time)

    def format_text(self) -> str:
        prefix = _SEVERITY_PREFIX.get(self.severity, "[ALERT]")
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return f"{prefix} {self.title}\n{self.message}\n-- {self.source} @ {time_str}"


class Notifier(ABC):
    name: str = "base"

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        raise NotImplementedError
