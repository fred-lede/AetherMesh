from __future__ import annotations

from runtime.security.rate_limiter import RateLimiter
from runtime.security.input_validator import InputValidator, ValidationError
from runtime.security.api_key_auth import APIKeyAuth


def test_rate_limiter_allows_within_burst() -> None:
    rl = RateLimiter(default_rate=100, default_burst=5)
    for _ in range(5):
        assert rl.check("key1") is True


def test_rate_limiter_exceeds_burst() -> None:
    rl = RateLimiter(default_rate=100, default_burst=3)
    for _ in range(3):
        rl.check("key2")
    assert rl.check("key2") is False


def test_rate_limiter_get_remaining() -> None:
    rl = RateLimiter(default_rate=100, default_burst=5)
    rl.check("key3")
    rl.check("key3")
    remaining = rl.get_remaining("key3")
    assert remaining > 0


def test_rate_limiter_reset() -> None:
    rl = RateLimiter(default_rate=100, default_burst=2)
    rl.check("k")
    rl.check("k")
    rl.reset("k")
    assert rl.check("k") is True


def test_rate_limiter_clear() -> None:
    rl = RateLimiter(default_rate=100, default_burst=2)
    rl.check("k")
    rl.clear()
    assert rl.get_remaining("k") == 2.0


def test_rate_limiter_independent_keys() -> None:
    rl = RateLimiter(default_rate=100, default_burst=2)
    assert rl.check("k1") is True
    assert rl.check("k1") is True
    assert rl.check("k1") is False
    assert rl.check("k2") is True


def test_input_validator_accepts_valid_text() -> None:
    v = InputValidator()
    result = v.validate_text("hello world", "prompt")
    assert result == "hello world"


def test_input_validator_rejects_oversized_text() -> None:
    v = InputValidator()
    try:
        v.validate_text("x" * (v.MAX_TEXT_LENGTH + 1), "big")
        assert False, "should have raised"
    except ValidationError as e:
        assert "exceeds max length" in str(e)


def test_input_validator_sanitizes_control_chars() -> None:
    v = InputValidator()
    result = v.sanitize("hello\x00world\x1f")
    assert result == "helloworld"


def test_input_validator_validates_messages() -> None:
    v = InputValidator()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    result = v.validate_messages(msgs)
    assert len(result) == 2


def test_input_validator_rejects_invalid_tool_name() -> None:
    v = InputValidator()
    try:
        v.validate_tool_name("invalid tool!")
        assert False, "should have raised"
    except ValidationError:
        pass


def test_input_validator_accepts_valid_tool_name() -> None:
    v = InputValidator()
    assert v.validate_tool_name("my_tool_1") == "my_tool_1"


def test_api_key_auth_disabled_by_default() -> None:
    a = APIKeyAuth()
    assert a.enabled is False


def test_api_key_auth_validate() -> None:
    a = APIKeyAuth()
    a.add_key("sk-test")
    assert a.validate("sk-test") is True
    assert a.validate("wrong") is False


def test_api_key_auth_add_remove() -> None:
    a = APIKeyAuth()
    a.add_key("sk-1")
    a.add_key("sk-2")
    assert a.snapshot()["key_count"] == 2
    a.remove_key("sk-1")
    assert a.validate("sk-1") is False
    assert a.validate("sk-2") is True
