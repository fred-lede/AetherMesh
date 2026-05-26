from __future__ import annotations

from providers.registry import Capability, parse_capabilities
from runtime.orchestration.capabilities import required_openai_capabilities, required_anthropic_capabilities


def test_documents_capability_enum():
    assert Capability.DOCUMENTS == "documents"
    assert Capability.DOCUMENTS.value == "documents"


def test_parse_capabilities_documents():
    caps = parse_capabilities(["chat", "documents"])
    assert Capability.DOCUMENTS in caps


def test_parse_capabilities_document_alias():
    caps = parse_capabilities(["document"])
    assert Capability.DOCUMENTS in caps


def test_parse_capabilities_file_alias():
    caps = parse_capabilities(["file"])
    assert Capability.DOCUMENTS in caps


def test_required_openai_capabilities_file_part():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "file", "file_id": "file_abc"}]}
        ]
    }
    caps = required_openai_capabilities(payload)
    assert "documents" in caps


def test_required_openai_capabilities_input_file_part():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "input_file", "file_id": "file_abc"}]}
        ]
    }
    caps = required_openai_capabilities(payload)
    assert "documents" in caps


def test_required_anthropic_capabilities_file_id_part():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "file_id", "file_id": "file_abc"}]}
        ]
    }
    caps = required_anthropic_capabilities(payload)
    assert "documents" in caps


def test_required_capabilities_no_documents_for_text():
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    caps = required_openai_capabilities(payload)
    assert "documents" not in caps
