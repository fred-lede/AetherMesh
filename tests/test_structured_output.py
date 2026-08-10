from __future__ import annotations

import json

import pytest

from runtime.orchestration.structured_output import (
    StructuredOutputError,
    apply_structured_output,
    build_repair_messages,
    extract_json_content,
    response_format_schema,
    validate_json,
)

SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, payload):
        self.calls.append(payload)
        return self.responses.pop(0)


def schema_payload(extra=None):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}},
    }
    if extra:
        payload.update(extra)
    return payload


def test_response_format_schema_json_schema():
    info = response_format_schema(schema_payload())
    assert info["kind"] == "schema"
    assert info["schema"] == SCHEMA
    assert info["name"] == "person"


def test_response_format_schema_json_object():
    info = response_format_schema({"response_format": {"type": "json_object"}})
    assert info["kind"] == "json_object"


def test_response_format_schema_none():
    assert response_format_schema({}) is None
    assert response_format_schema({"response_format": "text"}) is None


def test_extract_json_content_plain():
    assert extract_json_content('{"a": 1}') == {"a": 1}


def test_extract_json_content_fenced():
    assert extract_json_content('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_content_embedded():
    assert extract_json_content('here is the result: {"a": [1, 2, 3]} thanks') == {"a": [1, 2, 3]}


def test_extract_json_content_invalid():
    with pytest.raises(ValueError):
        extract_json_content("no json here")


def test_validate_json_valid():
    ok, err = validate_json({"name": "bob", "age": 3, "tags": ["x"]}, SCHEMA)
    assert ok
    assert err == ""


def test_validate_json_missing_required():
    ok, err = validate_json({"name": "bob"}, SCHEMA)
    assert not ok
    assert "age" in err


def test_validate_json_bad_type():
    ok, err = validate_json({"name": "bob", "age": "x"}, SCHEMA)
    assert not ok
    assert "age" in err


def test_validate_json_minimum():
    ok, err = validate_json({"name": "bob", "age": -1}, SCHEMA)
    assert not ok
    assert "minimum" in err


def test_validate_json_items():
    ok, err = validate_json({"name": "bob", "age": 1, "tags": [1]}, SCHEMA)
    assert not ok
    assert "tags" in err


def test_validate_json_enum():
    schema = {"type": "string", "enum": ["a", "b"]}
    assert validate_json("a", schema) == (True, "")
    assert validate_json("c", schema)[0] is False


def test_build_repair_messages_preserves_history():
    messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bad"},
    ]
    repaired = build_repair_messages(messages, {"schema": SCHEMA}, "bad", "invalid JSON")
    assert repaired[0]["role"] == "system"
    content = repaired[0]["content"]
    assert json.loads(content[content.index("{") :]) == SCHEMA
    assert [m["role"] for m in repaired[1:-1]] == ["user", "assistant"]
    assert repaired[-1]["role"] == "user"


def test_apply_structured_output_no_schema():
    adapter = FakeAdapter([{"choices": [{"message": {"content": "whatever"}}]}])
    payload = {"messages": []}
    response = apply_structured_output(payload, {"choices": [{"message": {"content": "whatever"}}]}, adapter)
    assert response["choices"][0]["message"]["content"] == "whatever"
    assert adapter.calls == []


def test_apply_structured_output_valid_content_normalized():
    adapter = FakeAdapter([])
    payload = schema_payload()
    response = {"choices": [{"message": {"content": '{"name": "bob", "age": 3}'}}]}
    result = apply_structured_output(payload, response, adapter)
    assert json.loads(result["choices"][0]["message"]["content"]) == {"name": "bob", "age": 3}
    assert adapter.calls == []


def test_apply_structured_output_repairs_invalid():
    adapter = FakeAdapter([{"choices": [{"message": {"content": '{"name": "bob", "age": 4}'}}]}])
    payload = schema_payload()
    response = {"choices": [{"message": {"content": "not json at all"}}]}
    result = apply_structured_output(payload, response, adapter)
    assert result["choices"][0]["message"]["content"] == json.dumps({"name": "bob", "age": 4}, ensure_ascii=False)
    assert len(adapter.calls) == 1
    assert "response_format" not in adapter.calls[0]


def test_apply_structured_output_repair_strips_response_format():
    adapter = FakeAdapter([{"choices": [{"message": {"content": '{"name": "bob", "age": 4}'}}]}])
    payload = schema_payload()
    apply_structured_output(payload, {"choices": [{"message": {"content": "garbage"}}]}, adapter)
    assert "response_format" not in adapter.calls[0]


def test_apply_structured_output_fails_after_retries():
    invalid = {"choices": [{"message": {"content": '{"name": "bob"}'}}]}
    adapter = FakeAdapter([invalid, invalid, invalid])
    payload = schema_payload()
    with pytest.raises(StructuredOutputError):
        apply_structured_output(payload, {"choices": [{"message": {"content": "garbage"}}]}, adapter)
    assert len(adapter.calls) == 3


def test_apply_structured_output_json_object_mode():
    adapter = FakeAdapter([{"choices": [{"message": {"content": '{"ok": true}'}}]}])
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }
    response = {"choices": [{"message": {"content": "plain text"}}]}
    result = apply_structured_output(payload, response, adapter)
    assert json.loads(result["choices"][0]["message"]["content"]) == {"ok": True}
    assert "response_format" not in adapter.calls[0]
