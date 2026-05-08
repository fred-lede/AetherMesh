from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("security.firewall")

SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore all previous instructions", re.IGNORECASE),
    re.compile(r"you are (now|not) (an? )?(AI|assistant|system)", re.IGNORECASE),
    re.compile(r"forget (all )?(previous|prior) (instructions|directions)", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"reveal your (system )?prompt", re.IGNORECASE),
    re.compile(r"output your (system )?(prompt|instructions)", re.IGNORECASE),
    re.compile(r"act as (if|though)", re.IGNORECASE),
    re.compile(r"do not follow (the )?(above|previous)", re.IGNORECASE),
    re.compile(r"you are required to", re.IGNORECASE),
]


class PromptFirewall:
    def __init__(self, patterns: list[re.Pattern] | None = None) -> None:
        self._patterns = patterns or SUSPICIOUS_PATTERNS

    def check_message(self, content: str) -> tuple[bool, str]:
        if not content:
            return True, ""
        for pattern in self._patterns:
            match = pattern.search(content)
            if match:
                logger.warning("Prompt firewall triggered: %r in %r", match.group(), content[:100])
                return False, f"Suspicious pattern detected: {match.group()}"
        return True, ""

    def check_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                passed, reason = self.check_message(content)
                if not passed:
                    warnings.append({"index": i, "role": msg.get("role", ""), "reason": reason})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        passed, reason = self.check_message(str(part.get("text", "")))
                        if not passed:
                            warnings.append({"index": i, "role": msg.get("role", ""), "reason": reason})
        return warnings


prompt_firewall = PromptFirewall()
