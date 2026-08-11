from __future__ import annotations

from runtime.orchestration.anthropic_converter import AnthropicRouter


def make_payload(**overrides):
    payload = {"model": "anthropic/claude-3-7-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return payload


def test_system_string_unchanged():
    router = AnthropicRouter()
    out = router._to_openai_payload(make_payload(system="be brief"))
    assert out["messages"][0] == {"role": "system", "content": "be brief"}


def test_system_list_without_cache_flattens():
    router = AnthropicRouter()
    out = router._to_openai_payload(make_payload(system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
    assert out["messages"][0] == {"role": "system", "content": "a\nb"}


def test_system_cache_control_preserved_as_array():
    router = AnthropicRouter()
    system = [
        {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "second block"},
    ]
    out = router._to_openai_payload(make_payload(system=system))
    system_msg = out["messages"][0]
    assert system_msg["role"] == "system"
    assert isinstance(system_msg["content"], list)
    assert system_msg["content"][0] == {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}}
    assert "cache_control" not in system_msg["content"][1]


def test_user_part_cache_control_preserved():
    router = AnthropicRouter()
    payload = make_payload(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "big context", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "question"},
                ],
            }
        ]
    )
    out = router._to_openai_payload(payload)
    parts = out["messages"][0]["content"]
    assert parts[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in parts[1]


def test_system_with_mixed_string_and_dict_blocks():
    router = AnthropicRouter()
    system = ["plain", {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}]
    out = router._to_openai_payload(make_payload(system=system))
    content = out["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "plain"}
    assert content[1]["cache_control"] == {"type": "ephemeral"}
