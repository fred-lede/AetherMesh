from __future__ import annotations

from unittest.mock import MagicMock, patch

from providers.nvidia_nim_adapter import NvidiaNIMAdapter


def _make_adapter() -> NvidiaNIMAdapter:
    with patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "test-key"}, clear=False):
        return NvidiaNIMAdapter()


def test_chat_drops_non_function_tools() -> None:
    adapter = _make_adapter()
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        adapter.chat({
            "model": "nvidia_nim/deepseek-r1",
            "tools": [
                {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}},
                {"type": "web_search", "max_results": 5},
                {"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}},
            ],
        })

    sent = post.call_args.args[1]
    assert sent["tools"] == [
        {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}},
    ]


def test_stream_drops_non_function_tools() -> None:
    adapter = _make_adapter()
    response = MagicMock(ok=True)
    response.iter_lines.return_value = ["data: [DONE]"]
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        list(adapter.stream({
            "model": "nvidia_nim/deepseek-r1",
            "tools": [
                {"type": "web_search", "max_results": 5},
                {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}},
            ],
        }))

    sent = post.call_args.args[1]
    assert sent["tools"] == [
        {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}},
    ]


def test_chat_without_tools_unchanged() -> None:
    adapter = _make_adapter()
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        adapter.chat({"model": "nvidia_nim/deepseek-r1", "messages": []})

    sent = post.call_args.args[1]
    assert "tools" not in sent


def test_chat_strips_responses_only_params() -> None:
    adapter = _make_adapter()
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        adapter.chat({
            "model": "nvidia_nim/deepseek-r1",
            "messages": [{"role": "user", "content": "hi"}],
            "include": ["reasoning.summary"],
            "reasoning": {"effort": "low"},
            "client_metadata": {"foo": "bar"},
            "prompt_cache_key": "abc",
            "temperature": 0.7,
            "max_tokens": 100,
        })

    sent = post.call_args.args[1]
    assert sent == {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }


def test_stream_strips_responses_only_params() -> None:
    adapter = _make_adapter()
    response = MagicMock(ok=True)
    response.iter_lines.return_value = ["data: [DONE]"]
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        list(adapter.stream({
            "model": "nvidia_nim/deepseek-r1",
            "messages": [{"role": "user", "content": "hi"}],
            "include": ["reasoning.summary"],
            "client_metadata": {"foo": "bar"},
            "prompt_cache_key": "abc",
            "temperature": 0.3,
        }))

    sent = post.call_args.args[1]
    assert "include" not in sent
    assert "client_metadata" not in sent
    assert "prompt_cache_key" not in sent
    assert "reasoning" not in sent
    assert sent["temperature"] == 0.3


def test_chat_sanitizes_and_restores_namespaced_tool_names() -> None:
    adapter = _make_adapter()
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "mcp__codegraph__codegraph_explore", "arguments": "{}"}},
    ]}}]}
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        result = adapter.chat({
            "model": "nvidia_nim/deepseek-r1",
            "tools": [
                {"type": "function", "function": {"name": "mcp__codegraph.codegraph_explore", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}},
            ],
        })

    sent = post.call_args.args[1]
    assert [t["function"]["name"] for t in sent["tools"]] == [
        "mcp__codegraph__codegraph_explore",
        "web_fetch",
    ]
    restored = result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    assert restored == "mcp__codegraph.codegraph_explore"


def test_stream_sanitizes_and_restores_namespaced_tool_names() -> None:
    adapter = _make_adapter()
    chunk = {
        "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "codegraph__codegraph_explore", "arguments": ""}},
        ]}}],
    }
    response = MagicMock(ok=True)
    response.iter_lines.return_value = [f"data: {__import__('json').dumps(chunk)}", "data: [DONE]"]
    with patch.object(adapter, "_post_with_retry", return_value=response) as post:
        yielded = [c for c in adapter.stream({
            "model": "nvidia_nim/deepseek-r1",
            "tools": [
                {"type": "function", "function": {"name": "codegraph.codegraph_explore", "parameters": {"type": "object"}}},
            ],
        })]

    sent = post.call_args.args[1]
    assert sent["tools"][0]["function"]["name"] == "codegraph__codegraph_explore"
    assert yielded[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "codegraph.codegraph_explore"
    assert yielded[1] == "[DONE]"


def test_sanitize_tool_name_preserves_valid_chars() -> None:
    adapter = _make_adapter()
    assert adapter._sanitize_tool_name("web_search") == "web_search"
    assert adapter._sanitize_tool_name("my-tool-1") == "my-tool-1"
    assert adapter._sanitize_tool_name("a.b.c") == "a__b__c"
    assert adapter._sanitize_tool_name("mcp__codegraph.codegraph_explore") == "mcp__codegraph__codegraph_explore"
