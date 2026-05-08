from __future__ import annotations

import re
from typing import Any


class ValidationError(Exception):
    pass


class InputValidator:
    MAX_TEXT_LENGTH = 100_000
    MAX_MESSAGES = 100
    MAX_TOOL_NAME_LENGTH = 128

    def __init__(self) -> None:
        self._control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def validate_text(self, text: str, field_name: str = "text") -> str:
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string")
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValidationError(
                f"{field_name} exceeds max length of {self.MAX_TEXT_LENGTH}"
            )
        return self.sanitize(text)

    def validate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            raise ValidationError("messages must be a list")
        if len(messages) > self.MAX_MESSAGES:
            raise ValidationError(f"messages exceeds max count of {self.MAX_MESSAGES}")
        validated: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValidationError("each message must be an object")
            content = msg.get("content", "")
            role = msg.get("role", "")
            if isinstance(content, str):
                msg["content"] = self.validate_text(content, "message.content")
            validated.append(msg)
        return validated

    def validate_tool_name(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("tool name must be a non-empty string")
        if len(name) > self.MAX_TOOL_NAME_LENGTH:
            raise ValidationError(f"tool name exceeds max length of {self.MAX_TOOL_NAME_LENGTH}")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValidationError(f"invalid tool name: {name}")
        return name.strip()

    def sanitize(self, text: str) -> str:
        return self._control_chars.sub("", text)


input_validator = InputValidator()
