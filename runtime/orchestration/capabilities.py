from __future__ import annotations

from typing import Any


def required_openai_capabilities(payload: dict[str, Any]) -> set[str]:
    required = {"chat"}
    if payload.get("tools"):
        required.add("tools")
    if payload.get("thinking") or payload.get("reasoning"):
        required.add("thinking")

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            _add_content_capabilities(required, message.get("content"))
    return required


def required_anthropic_capabilities(payload: dict[str, Any]) -> set[str]:
    required = {"chat"}
    if payload.get("tools"):
        required.add("tools")
    if payload.get("thinking"):
        required.add("thinking")

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            _add_content_capabilities(required, message.get("content"))
    return required


def _add_content_capabilities(required: set[str], content: Any) -> None:
    if isinstance(content, dict):
        _add_part_capability(required, content)
        return
    if not isinstance(content, list):
        return
    for part in content:
        if isinstance(part, dict):
            _add_part_capability(required, part)


def _add_part_capability(required: set[str], part: dict[str, Any]) -> None:
    part_type = str(part.get("type", "")).lower()
    if part_type in {"image", "image_url", "input_image"}:
        required.add("vision")
    elif part_type in {"audio", "input_audio"}:
        required.add("audio")
    elif part_type == "tool_result":
        _add_tool_result_nested_capabilities(required, part.get("content"))
    elif part_type == "thinking":
        required.add("thinking")


def _add_tool_result_nested_capabilities(required: set[str], content: Any) -> None:
    if isinstance(content, dict):
        _add_part_capability(required, content)
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, dict):
            _add_part_capability(required, item)
