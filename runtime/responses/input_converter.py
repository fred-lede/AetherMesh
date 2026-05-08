from __future__ import annotations

from typing import Any

from runtime.responses.response_models import InputItem, InputItemType


def responses_input_to_messages(
    input_value: Any,
    instructions: str = "",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if isinstance(input_value, dict):
        item = _parse_input_item(input_value)
        messages.extend(_input_item_to_messages(item))
        return messages

    if isinstance(input_value, list):
        for item_data in input_value:
            if isinstance(item_data, str):
                messages.append({"role": "user", "content": item_data})
            elif isinstance(item_data, dict):
                item = _parse_input_item(item_data)
                messages.extend(_input_item_to_messages(item))

    return messages


def _parse_input_item(item: dict[str, Any]) -> InputItem:
    item_type_str = str(item.get("type", "message"))
    try:
        item_type = InputItemType(item_type_str)
    except ValueError:
        item_type = InputItemType.MESSAGE

    result = InputItem(
        id=str(item.get("id", f"input_{id(item)}")),
        type=item_type,
        role=str(item.get("role", "user")),
    )

    if item_type == InputItemType.MESSAGE:
        content = item.get("content", "")
        if isinstance(content, str):
            result.content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            result.content = content
        else:
            result.content = [{"type": "text", "text": str(content)}]

    elif item_type == InputItemType.TOOL_CALL:
        result.tool_call_id = str(item.get("tool_call_id", ""))
        result.tool_name = str(item.get("tool_name", ""))
        result.arguments = _stringify_arguments(item.get("arguments", "{}"))

    elif item_type == InputItemType.TOOL_RESULT:
        result.tool_call_id = str(item.get("tool_call_id", ""))
        result.output = str(item.get("output", ""))
        result.is_error = bool(item.get("is_error", False))

    elif item_type == InputItemType.FILE:
        result.file_id = str(item.get("file_id", ""))
        result.filename = str(item.get("filename", ""))

    return result


def _input_item_to_messages(item: InputItem) -> list[dict[str, Any]]:
    if item.type == InputItemType.MESSAGE:
        content = _content_list_to_various(item.content)
        return [{"role": item.role, "content": content}]

    if item.type == InputItemType.TOOL_CALL:
        return [{
            "role": "assistant",
            "tool_calls": [{
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.tool_name,
                    "arguments": item.arguments,
                },
            }],
        }]

    if item.type == InputItemType.TOOL_RESULT:
        return [{
            "role": "tool",
            "tool_call_id": item.tool_call_id,
            "content": item.output,
        }]

    if item.type == InputItemType.FILE:
        return [{
            "role": "user",
            "content": f"[file: {item.filename or item.file_id}]",
        }]

    return [{"role": "user", "content": str(item)}]


def _content_list_to_various(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0].get("text", ""))
    return content


def _stringify_arguments(args: Any) -> str:
    if isinstance(args, str):
        return args
    import json
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


def chat_completion_to_response_messages(
    completion: dict[str, Any],
    model: str,
    previous_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result = list(previous_messages or [])
    choices = completion.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        result.append(message)
    return result
