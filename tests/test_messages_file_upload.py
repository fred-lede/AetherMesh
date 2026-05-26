from __future__ import annotations

from unittest.mock import patch

from router.anthropic.messages_adapter import _resolve_file_content_blocks


def test_no_file_id_blocks_unchanged() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
    }
    result, file_ids = _resolve_file_content_blocks(payload)
    assert file_ids == []
    assert result["messages"][0]["content"][0]["text"] == "Hello"


def test_file_id_block_resolved() -> None:
    fake_resolved = [{"type": "text", "text": "Resolved content"}]
    with patch("router.anthropic.messages_adapter.resolve_file_blocks", return_value=fake_resolved) as mock_r:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in"},
                        {"type": "file_id", "file_id": "file_abc123"},
                    ],
                }
            ]
        }
        result, file_ids = _resolve_file_content_blocks(payload)
        assert file_ids == ["file_abc123"]
        blocks = result["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["text"] == "What's in"
        assert blocks[1]["text"] == "Resolved content"
        mock_r.assert_called_once_with(["file_abc123"], "generic", None)


def test_multiple_file_ids_across_messages() -> None:
    fake = [{"type": "text", "text": "R"}]
    with patch("router.anthropic.messages_adapter.resolve_file_blocks", return_value=fake):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "file_id", "file_id": "file_a"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
                {"role": "user", "content": [{"type": "file_id", "file_id": "file_b"}]},
            ]
        }
        result, file_ids = _resolve_file_content_blocks(payload)
        assert file_ids == ["file_a", "file_b"]
