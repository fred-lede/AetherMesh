from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("security.secret")

SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(secret|password|token|bearer)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)ghp_[A-Za-z0-9]{36}"),
    re.compile(r"(?i)gho_[A-Za-z0-9]{36}"),
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
]


class SecretDetector:
    def __init__(self, patterns: list[re.Pattern] | None = None) -> None:
        self._patterns = patterns or SECRET_PATTERNS

    def contains_secret(self, text: str) -> bool:
        if not text:
            return False
        for pattern in self._patterns:
            if pattern.search(text):
                return True
        return False

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        result = text
        for pattern in self._patterns:
            result = pattern.sub(replacement, result)
        return result

    def check_output(self, output: str) -> tuple[bool, str]:
        if self.contains_secret(output):
            logger.warning("Secret detected in output, redacting")
            return False, self.redact(output)
        return True, output


secret_detector = SecretDetector()
