from __future__ import annotations

from runtime.responses.input_converter import _truncate_input_list, responses_input_to_messages


def _system_input(count: int) -> list[dict]:
    return [
        {"type": "message", "role": "system", "content": f"sys-{i}"}
        for i in range(count)
    ]


def test_truncation_preserves_user_message_after_large_tool_turn():
    input_value = _system_input(11)
    input_value.append({"type": "message", "role": "user", "content": "please do it"})
    input_value.append({
        "type": "function_call",
        "tool_call_id": "call_0",
        "name": "exec_command",
        "arguments": "{}",
    })
    input_value.extend(
        {"type": "function_call_output", "call_id": f"call_{i}", "output": "done"}
        for i in range(7)
    )

    messages = responses_input_to_messages(input_value)

    assert any(m.get("role") == "user" for m in messages)


def test_truncation_preserves_user_when_input_has_no_tool_results():
    input_value = _system_input(11)
    input_value.append({"type": "message", "role": "user", "content": "hello"})
    input_value.append({"type": "message", "role": "assistant", "content": "hi"})

    messages = responses_input_to_messages(input_value)

    assert any(m.get("role") == "user" for m in messages)


def test_truncation_with_no_user_returns_unchanged():
    input_value = _system_input(3)
    input_value.append({"type": "function_call_output", "call_id": "call_0", "output": "x"})

    messages = responses_input_to_messages(input_value)

    assert messages == [
        {"role": "system", "content": "sys-0"},
        {"role": "system", "content": "sys-1"},
        {"role": "system", "content": "sys-2"},
        {"role": "tool", "tool_call_id": "call_0", "content": "x"},
    ]


def test_truncate_input_list_keeps_last_user_when_dropped():
    parsed = [
        {"role": "system", "content": f"s{i}"} for i in range(11)
    ]
    parsed.append({"role": "user", "content": "please do it"})
    parsed.append({
        "role": "assistant",
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "exec_command", "arguments": "{}"}}],
    })
    parsed.extend({"role": "tool", "tool_call_id": f"call_{i}", "content": "done"} for i in range(7))

    result = _truncate_input_list(parsed)

    assert any(m.get("role") == "user" for m in result)
