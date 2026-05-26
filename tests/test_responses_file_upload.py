from __future__ import annotations

from unittest.mock import patch

from runtime.responses.input_converter import responses_input_to_messages


def test_file_input_resolved():
    fake_resolved = [{"type": "text", "text": "File content here"}]
    with patch("runtime.responses.input_converter.resolve_file_blocks", return_value=fake_resolved) as mock_r:
        messages = responses_input_to_messages([
            {"type": "file", "file_id": "file_abc123", "filename": "test.pdf"},
        ])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "File content here"
        mock_r.assert_called_once_with(["file_abc123"], "generic")


def test_file_input_fallback_on_empty_resolve():
    with patch("runtime.responses.input_converter.resolve_file_blocks", return_value=[]):
        messages = responses_input_to_messages([
            {"type": "file", "file_id": "file_missing", "filename": "gone.txt"},
        ])
        assert len(messages) == 1
        assert "[file: gone.txt]" in str(messages[0]["content"])


def test_file_input_with_message():
    fake_resolved = [{"type": "text", "text": "Parsed doc"}]
    with patch("runtime.responses.input_converter.resolve_file_blocks", return_value=fake_resolved):
        messages = responses_input_to_messages([
            {"type": "message", "role": "user", "content": "Analyze this"},
            {"type": "file", "file_id": "file_doc1"},
        ])
        assert len(messages) == 2
        assert messages[0]["content"] == "Analyze this"
        assert messages[1]["content"] == "Parsed doc"


def test_file_input_as_input_file_type():
    fake_resolved = [{"type": "text", "text": "Resolved"}]
    with patch("runtime.responses.input_converter.resolve_file_blocks", return_value=fake_resolved):
        messages = responses_input_to_messages([
            {"type": "input_file", "file_id": "file_xyz"},
        ])
        assert len(messages) == 1
        assert messages[0]["content"] == "Resolved"
