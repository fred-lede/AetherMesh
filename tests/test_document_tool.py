from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import settings
from runtime.documents.mineru_converter import MinerUError, convert_document, mineru_available
from runtime.tools.builtin.document import DOCUMENT_DESCRIPTOR, _document_handler
from runtime.tools.tool_result import ToolCall


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_document_tool_registered() -> None:
    assert DOCUMENT_DESCRIPTOR.name == "document_to_markdown"
    assert "path" in DOCUMENT_DESCRIPTOR.input_schema["properties"]
    assert DOCUMENT_DESCRIPTOR.requires_confirmation is True


def test_document_tool_missing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mineru_enabled", True)
    result = _document_handler(ToolCall(id="1", name="document_to_markdown", arguments={}))
    assert result.is_error is True
    assert "path" in str(result.output)


def test_document_tool_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mineru_enabled", False)
    result = _document_handler(
        ToolCall(id="1", name="document_to_markdown", arguments={"path": "x.pdf"})
    )
    assert result.is_error is True
    assert "disabled" in str(result.output)


def test_document_tool_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "mineru_enabled", True)

    def fake_convert(file_path, out_dir=None, backend=None, method=None, timeout_s=None):
        return {
            "markdown": "# Hello\n\nextracted body",
            "source": str(file_path),
            "output_path": str(tmp_path / "x.md"),
            "duration_ms": 123,
            "chars": 40,
        }

    monkeypatch.setattr("runtime.documents.mineru_converter.convert_document", fake_convert)
    result = _document_handler(
        ToolCall(id="1", name="document_to_markdown", arguments={"path": "report.pdf"})
    )
    assert result.is_error is False
    assert "extracted body" in str(result.output)
    assert result.metadata["output_path"].endswith("x.md")


def test_document_tool_mineru_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mineru_enabled", True)

    def raise_error(*args, **kwargs):
        raise MinerUError("MinerU exited with code 1")

    monkeypatch.setattr("runtime.documents.mineru_converter.convert_document", raise_error)
    result = _document_handler(
        ToolCall(id="1", name="document_to_markdown", arguments={"path": "bad.pdf"})
    )
    assert result.is_error is True
    assert "code 1" in str(result.output)


def test_convert_document_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MinerUError):
        convert_document(tmp_path / "nope.pdf")


def test_convert_document_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    out = tmp_path / "out"
    (out / "doc").mkdir(parents=True)
    (out / "doc" / "doc.md").write_text("# converted", encoding="utf-8")
    monkeypatch.setattr(
        "runtime.documents.mineru_converter._mineru_command",
        lambda: [str(tmp_path / "python.exe"), "-m", "mineru.cli.client"],
    )
    monkeypatch.setattr(
        settings,
        "mineru_python",
        "",
    )
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = convert_document(source, out_dir=out)
    assert result["markdown"] == "# converted"
    assert result["chars"] == 11
    assert captured[0][0].endswith("python.exe")
    assert captured[0][1] == "-m"
    assert captured[0][2] == "mineru.cli.client"
    assert "-p" in captured[0]


def test_convert_document_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(
        "runtime.documents.mineru_converter._mineru_command",
        lambda: [str(tmp_path / "python.exe"), "-m", "mineru.cli.client"],
    )
    monkeypatch.setattr(settings, "mineru_python", "")

    def fake_run(cmd, **kwargs):
        return _Result(returncode=2, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(MinerUError, match="code 2"):
        convert_document(source, out_dir=tmp_path / "out")


def test_mineru_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "runtime.documents.mineru_converter._mineru_command",
        lambda: [str(tmp_path / "python.exe"), "-m", "mineru.cli.client"],
    )
    assert mineru_available() is True
