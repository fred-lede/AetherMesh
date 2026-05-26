from __future__ import annotations

from unittest.mock import patch

from runtime.orchestration.openai_handler import _resolve_file_ids_in_payload


def test_no_file_refs_unchanged():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
    }
    result, file_ids = _resolve_file_ids_in_payload(payload)
    assert file_ids == []
    assert result["messages"][0]["content"][0]["text"] == "Hello"


def test_single_file_ref_resolved():
    fake_resolved = [{"type": "text", "text": "Resolved content"}]
    with patch("runtime.orchestration.openai_handler.resolve_file_blocks", return_value=fake_resolved) as mock_r:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this: "},
                        {"type": "file", "file_id": "file_abc123"},
                    ],
                }
            ]
        }
        result, file_ids = _resolve_file_ids_in_payload(payload)
        assert file_ids == ["file_abc123"]
        blocks = result["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["text"] == "Analyze this: "
        assert blocks[1] == fake_resolved[0]
        mock_r.assert_called_once_with(["file_abc123"], "generic")


def test_multiple_file_refs():
    fake = [{"type": "text", "text": "R"}]
    with patch("runtime.orchestration.openai_handler.resolve_file_blocks", return_value=fake):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "file", "file_id": "file_a"}]},
                {"role": "user", "content": [{"type": "file", "file_id": "file_b"}]},
            ]
        }
        result, file_ids = _resolve_file_ids_in_payload(payload)
        assert file_ids == ["file_a", "file_b"]


def test_string_content_unchanged():
    payload = {
        "messages": [
            {"role": "user", "content": "Hello string content"}
        ]
    }
    result, file_ids = _resolve_file_ids_in_payload(payload)
    assert file_ids == []
    assert result["messages"][0]["content"] == "Hello string content"


def test_file_id_type_resolved():
    fake = [{"type": "text", "text": "R"}]
    with patch("runtime.orchestration.openai_handler.resolve_file_blocks", return_value=fake):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "file_id", "file_id": "file_xyz"}]}
            ]
        }
        result, file_ids = _resolve_file_ids_in_payload(payload)
        assert file_ids == ["file_xyz"]
        assert result["messages"][0]["content"][0]["text"] == "R"
