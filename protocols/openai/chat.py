from __future__ import annotations

from typing import Any


def openai_message(role: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": role, "content": content}


def user_message(content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return openai_message("user", content)


def assistant_message(content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return openai_message("assistant", content)


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def tool_message(content: str, tool_call_id: str) -> dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def function_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_content(image_url: str, detail: str = "auto") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
