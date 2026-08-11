from __future__ import annotations

from runtime.orchestration.anthropic_converter import AnthropicRouter
from runtime.orchestration.openai_handler import RouterService

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
]


class TestAnthropicToolChoice:
    def setup_method(self) -> None:
        self.router = AnthropicRouter()

    def test_any_maps_to_required_string(self) -> None:
        assert self.router._anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_none_maps_to_none_string(self) -> None:
        assert self.router._anthropic_tool_choice_to_openai({"type": "none"}) == "none"

    def test_auto_maps_to_auto_string(self) -> None:
        assert self.router._anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_default_maps_to_auto_string(self) -> None:
        assert self.router._anthropic_tool_choice_to_openai({}) == "auto"

    def test_tool_with_name_maps_to_function_object(self) -> None:
        out = self.router._anthropic_tool_choice_to_openai({"type": "tool", "name": "get_weather"})
        assert out == {"type": "function", "function": {"name": "get_weather"}}

    def test_tool_without_name_maps_to_auto_string(self) -> None:
        assert self.router._anthropic_tool_choice_to_openai({"type": "tool"}) == "auto"


class TestNormalizeToolChoice:
    def setup_method(self) -> None:
        self.service = RouterService()

    def test_string_variants_kept(self) -> None:
        for value in ("auto", "none", "required"):
            assert self.service._normalize_tool_choice(value, TOOLS) == value

    def test_unknown_string_dropped(self) -> None:
        assert self.service._normalize_tool_choice("always", TOOLS) is None

    def test_function_dict_kept_when_name_exists(self) -> None:
        out = self.service._normalize_tool_choice({"type": "function", "function": {"name": "get_weather"}}, TOOLS)
        assert out == {"type": "function", "function": {"name": "get_weather"}}

    def test_function_dict_without_function_dropped(self) -> None:
        assert self.service._normalize_tool_choice({"type": "function"}, TOOLS) is None

    def test_function_dict_unknown_name_dropped(self) -> None:
        out = self.service._normalize_tool_choice({"type": "function", "function": {"name": "nope"}}, TOOLS)
        assert out is None

    def test_flat_name_field_normalized(self) -> None:
        out = self.service._normalize_tool_choice({"type": "function", "name": "get_time"}, TOOLS)
        assert out == {"type": "function", "function": {"name": "get_time"}}

    def test_typed_dict_variants_normalized_to_strings(self) -> None:
        assert self.service._normalize_tool_choice({"type": "required"}, TOOLS) == "required"
        assert self.service._normalize_tool_choice({"type": "none"}, TOOLS) == "none"
        assert self.service._normalize_tool_choice({"type": "auto"}, TOOLS) is None

    def test_empty_tools_drop(self) -> None:
        assert self.service._normalize_tool_choice("auto", []) is None
        assert self.service._normalize_tool_choice("auto", None) is None

    def test_non_dict_non_string_dropped(self) -> None:
        assert self.service._normalize_tool_choice(123, TOOLS) is None


class TestPayloadNormalization:
    def setup_method(self) -> None:
        self.service = RouterService()

    def test_custom_provider_keeps_valid_tool_choice(self) -> None:
        payload = {"model": "x", "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
        out = self.service._normalize_payload_for_provider(payload, "agnes")
        assert out["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_custom_provider_drops_malformed_tool_choice(self) -> None:
        payload = {"model": "x", "tools": TOOLS, "tool_choice": {"type": "function"}}
        out = self.service._normalize_payload_for_provider(payload, "agnes")
        assert "tool_choice" not in out

    def test_custom_provider_normalizes_typed_dict(self) -> None:
        payload = {"model": "x", "tools": TOOLS, "tool_choice": {"type": "required"}}
        out = self.service._normalize_payload_for_provider(payload, "agnes")
        assert out["tool_choice"] == "required"

    def test_no_tools_drops_tool_choice(self) -> None:
        payload = {"model": "x", "tool_choice": "auto"}
        out = self.service._normalize_payload_for_provider(payload, "agnes")
        assert "tool_choice" not in out
