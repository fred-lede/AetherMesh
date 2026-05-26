from __future__ import annotations

import base64
import json
from pathlib import Path
from tempfile import mkdtemp

import pytest

from runtime.tools.content_blocks import (
    anthropic_block_to_openai_parts,
    anthropic_content_to_openai_parts,
    content_part_to_text_and_images,
    resolve_file_blocks,
)


def test_anthropic_text_plain_document_decodes_to_text() -> None:
    data = base64.b64encode("hello document".encode("utf-8")).decode("ascii")

    parts = anthropic_block_to_openai_parts(
        {
            "type": "document",
            "title": "note.txt",
            "source": {"type": "base64", "media_type": "text/plain", "data": data},
        }
    )

    assert parts == [{"type": "text", "text": "[Document: note.txt]\nhello document"}]


def test_anthropic_pdf_document_uses_media_type_hint() -> None:
    parts = anthropic_block_to_openai_parts(
        {
            "type": "document",
            "title": "report.pdf",
            "source": {
                "type": "url",
                "media_type": "application/pdf",
                "url": "https://example.test/report.pdf",
            },
        }
    )

    assert parts == [
        {
            "type": "text",
            "text": "[Document: report.pdf (application/pdf), url=https://example.test/report.pdf]",
        }
    ]


def test_anthropic_image_converts_to_openai_image_url() -> None:
    parts = anthropic_block_to_openai_parts(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
            "detail": "high",
        }
    )

    assert parts == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123", "detail": "high"},
        }
    ]


def test_tool_result_preserves_text_and_nested_image() -> None:
    parts = anthropic_block_to_openai_parts(
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [
                {"type": "text", "text": "done"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "img"},
                },
            ],
        }
    )

    assert parts[0] == {"type": "text", "text": "[Tool Result (toolu_1)]: done"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,img", "detail": "auto"},
    }


def test_anthropic_audio_converts_to_openai_input_audio() -> None:
    parts = anthropic_block_to_openai_parts(
        {
            "type": "audio",
            "source": {"type": "base64", "media_type": "audio/wav", "data": "audio-data"},
        }
    )

    assert parts == [
        {
            "type": "input_audio",
            "input_audio": {"data": "audio-data", "format": "wav"},
        }
    ]


def test_openai_parts_flatten_for_ollama_text_images_and_audio_hint() -> None:
    text, images = content_part_to_text_and_images({"type": "text", "text": "hello"})
    assert text == "hello"
    assert images == []

    text, images = content_part_to_text_and_images(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    )
    assert text == ""
    assert images == ["abc"]

    text, images = content_part_to_text_and_images(
        {"type": "input_audio", "input_audio": {"data": "audio-data", "format": "wav"}}
    )
    assert text == "[Audio: wav]"
    assert images == []


def test_anthropic_content_to_openai_parts_preserves_order() -> None:
    parts = anthropic_content_to_openai_parts(
        [
            {"type": "text", "text": "before"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.test/a.png"},
            },
            {"type": "text", "text": "after"},
        ]
    )

    assert [part["type"] for part in parts] == ["text", "image_url", "text"]


@pytest.fixture
def upload_dir() -> Path:
    d = Path(mkdtemp())
    yield d
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def create_file_metadata(upload_dir: Path, file_id: str, mime_type: str, content_text: str) -> None:
    meta = {"mime_type": mime_type, "original_name": f"test.{mime_type.split('/')[-1]}"}
    (upload_dir / file_id).write_text(content_text)
    (upload_dir / f"{file_id}.meta.json").write_text(json.dumps(meta))


def test_resolve_anthropic_pdf(upload_dir: Path) -> None:
    file_id = "file_abc123"
    create_file_metadata(upload_dir, file_id, "application/pdf", "PDF text content")

    result = resolve_file_blocks([file_id], "anthropic", upload_dir)
    assert len(result) == 1
    block = result[0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["source"]["type"] == "base64"
    assert isinstance(block["source"]["data"], str)
    assert len(block["source"]["data"]) > 0


def test_resolve_anthropic_text_file(upload_dir: Path) -> None:
    file_id = "file_def456"
    create_file_metadata(upload_dir, file_id, "text/plain", "Hello world")

    result = resolve_file_blocks([file_id], "anthropic", upload_dir)
    assert len(result) == 1
    block = result[0]
    assert block["type"] == "text"
    assert block["text"] == "Hello world"


def test_resolve_openai_text_file(upload_dir: Path) -> None:
    file_id = "file_ghi789"
    create_file_metadata(upload_dir, file_id, "text/markdown", "# Title")

    result = resolve_file_blocks([file_id], "openai", upload_dir)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "# Title" in result[0]["text"]


def test_resolve_multiple_files(upload_dir: Path) -> None:
    ids = ["file_a", "file_b"]
    create_file_metadata(upload_dir, "file_a", "text/plain", "File A")
    create_file_metadata(upload_dir, "file_b", "text/plain", "File B")

    result = resolve_file_blocks(ids, "openai", upload_dir)
    assert len(result) == 2
    assert result[0]["text"] == "File A"
    assert result[1]["text"] == "File B"


def test_resolve_unknown_file_id_returns_empty(upload_dir: Path) -> None:
    result = resolve_file_blocks(["file_nonexistent"], "openai", upload_dir)
    assert result == []


def test_resolve_empty_file_ids(upload_dir: Path) -> None:
    result = resolve_file_blocks([], "openai", upload_dir)
    assert result == []


def test_resolve_gemini_text_file(upload_dir: Path) -> None:
    file_id = "file_gemini"
    create_file_metadata(upload_dir, file_id, "text/plain", "Gemini content")

    result = resolve_file_blocks([file_id], "gemini", upload_dir)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "Gemini content" in result[0]["text"]


def test_resolve_ollama_text_file(upload_dir: Path) -> None:
    file_id = "file_ollama"
    create_file_metadata(
        upload_dir,
        file_id,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "DOCX content",
    )

    result = resolve_file_blocks([file_id], "ollama", upload_dir)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "DOCX content" in result[0]["text"]
