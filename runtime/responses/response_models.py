from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseStatus(Enum):
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InputItemType(Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE = "file"


class OutputItemType(Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"


class ContentPartType(Enum):
    OUTPUT_TEXT = "output_text"
    REFUSAL = "refusal"
    REASONING = "reasoning"


@dataclass
class ResponseUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ContentPart:
    type: ContentPartType = ContentPartType.OUTPUT_TEXT
    text: str = ""
    annotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value}
        if self.type == ContentPartType.OUTPUT_TEXT:
            d["text"] = self.text
            d["annotations"] = self.annotations
        elif self.type == ContentPartType.REFUSAL:
            d["refusal"] = self.text
        elif self.type == ContentPartType.REASONING:
            d["text"] = self.text
        return d


@dataclass
class OutputItem:
    id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex[:16]}")
    type: OutputItemType = OutputItemType.MESSAGE
    role: str = "assistant"
    content: list[ContentPart] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    output: str = ""
    is_error: bool = False
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "status": self.status,
        }
        if self.type == OutputItemType.MESSAGE:
            d["role"] = self.role
            d["content"] = [p.to_dict() for p in self.content]
        elif self.type == OutputItemType.TOOL_CALL:
            d["tool_call_id"] = self.tool_call_id
            d["tool_name"] = self.tool_name
            d["arguments"] = self.arguments
        elif self.type == OutputItemType.REASONING:
            d["content"] = [p.to_dict() for p in self.content]
        return d


@dataclass
class InputItem:
    id: str = field(default_factory=lambda: f"input_{uuid.uuid4().hex[:16]}")
    type: InputItemType = InputItemType.MESSAGE
    role: str = "user"
    content: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    output: str = ""
    is_error: bool = False
    file_id: str = ""
    filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
        }
        if self.type == InputItemType.MESSAGE:
            d["role"] = self.role
            d["content"] = self.content
        elif self.type == InputItemType.TOOL_CALL:
            d["tool_call_id"] = self.tool_call_id
            d["tool_name"] = self.tool_name
            d["arguments"] = self.arguments
        elif self.type == InputItemType.TOOL_RESULT:
            d["tool_call_id"] = self.tool_call_id
            d["output"] = self.output
            d["is_error"] = self.is_error
        elif self.type == InputItemType.FILE:
            d["file_id"] = self.file_id
            if self.filename:
                d["filename"] = self.filename
        return d


@dataclass
class ResponseObject:
    id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:24]}")
    object: str = "response"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    status: ResponseStatus = ResponseStatus.IN_PROGRESS
    instructions: str = ""
    output: list[OutputItem] = field(default_factory=list)
    usage: ResponseUsage = field(default_factory=ResponseUsage)
    error: dict[str, Any] | None = None
    previous_response_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "status": self.status.value,
            "output": [item.to_dict() for item in self.output],
            "usage": self.usage.to_dict(),
        }
        if self.instructions:
            d["instructions"] = self.instructions
        if self.error:
            d["error"] = self.error
        if self.previous_response_id:
            d["previous_response_id"] = self.previous_response_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def make_text_output(text: str, role: str = "assistant") -> OutputItem:
    return OutputItem(
        type=OutputItemType.MESSAGE,
        role=role,
        content=[ContentPart(type=ContentPartType.OUTPUT_TEXT, text=text)],
    )


def make_text_content_part(text: str) -> ContentPart:
    return ContentPart(type=ContentPartType.OUTPUT_TEXT, text=text)


def make_tool_call_output(
    tool_call_id: str,
    tool_name: str,
    arguments: str,
) -> OutputItem:
    return OutputItem(
        type=OutputItemType.TOOL_CALL,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
