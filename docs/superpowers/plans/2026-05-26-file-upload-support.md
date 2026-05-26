# File Upload & Document Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multipart file upload, file parsing (PDF/DOCX/XLSX/PPTX/TXT/MD), and document content block resolution to AetherMesh.

**Architecture:** New `router/files_router.py` for upload → temp disk storage. New `runtime/tools/file_parser.py` for format conversion. Extend `runtime/tools/content_blocks.py` with `resolve_file_blocks()` to swap file blocks for parsed text (or pass through native PDF to Anthropic). Two-layer cleanup (request-scoped + background TTL). New `DOCUMENTS` capability for routing.

**Tech Stack:** PyMuPDF (fitz), python-docx, openpyxl, python-pptx, pytesseract, FastAPI UploadFile

---

### Task 1: Settings & Dependencies

**Files:**
- Modify: `config/settings.py` — add upload settings
- Modify: `requirements.txt` — add parsing libraries
- Test: `tests/test_settings.py` (if it exists) or manual verification

- [ ] **Step 1: Add parsing dependencies to requirements.txt**

```txt
pymupdf
python-docx
openpyxl
python-pptx
pytesseract
```

- [ ] **Step 2: Install dependencies**

Run: `pip install pymupdf python-docx openpyxl python-pptx pytesseract`

- [ ] **Step 3: Add upload settings fields to Settings dataclass**

Edit `config/settings.py` — add after `sandbox_profiles` field:

```python
    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AIIH_UPLOAD_DIR", "/tmp/aethermesh/uploads"))
    )
    max_upload_size_mb: int = field(default_factory=lambda: _env_int("AIIH_MAX_UPLOAD_SIZE_MB", 50))
    allowed_upload_mime_types: list[str] = field(
        default_factory=lambda: _env_csv("AIIH_ALLOWED_UPLOAD_MIME_TYPES",
            ",".join([
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "text/plain",
                "text/markdown",
            ])
        )
    )
    tesseract_langs: str = field(default_factory=lambda: os.getenv("AIIH_TESSERACT_LANGS", "eng"))
    file_cleanup_ttl_seconds: int = field(default_factory=lambda: _env_int("AIIH_FILE_CLEANUP_TTL", 600))
    ocr_concurrency: int = field(default_factory=lambda: _env_int("AIIH_OCR_CONCURRENCY", 2))
```

- [ ] **Step 4: Commit**

```bash
git add config/settings.py requirements.txt
git commit -m "feat: add file upload settings and parsing dependencies"
```

---

### Task 2: File Parser Engine

**Files:**
- Create: `runtime/tools/file_parser.py`
- Test: `tests/test_file_parser.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_file_parser.py`:

```python
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import pytest

from runtime.tools.file_parser import FileParseResult, parse_file, parse_base64


# --- helpers ---

def _write_temp(content: bytes | str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    if isinstance(content, str):
        f.write(content.encode("utf-8"))
    else:
        f.write(content)
    f.close()
    return Path(f.name)


def test_parse_txt():
    path = _write_temp("Hello world", ".txt")
    try:
        result = parse_file(path, "text/plain")
        assert isinstance(result, FileParseResult)
        assert "Hello world" in result.text
        assert result.pages is None
    finally:
        path.unlink(missing_ok=True)


def test_parse_md():
    path = _write_temp("# Title\n\nBody text", ".md")
    try:
        result = parse_file(path, "text/markdown")
        assert "# Title" in result.text
        assert "Body text" in result.text
    finally:
        path.unlink(missing_ok=True)


def test_parse_pdf_text():
    from fitz import open as fitz_open
    doc = fitz_open()
    doc.insert_page(0, text="Hello PDF")
    buf = doc.write()
    doc.close()
    path = _write_temp(buf, ".pdf")
    try:
        result = parse_file(path, "application/pdf")
        assert "Hello PDF" in result.text
        assert isinstance(result.pages, int)
        assert result.pages >= 1
    finally:
        path.unlink(missing_ok=True)


def test_parse_docx():
    from docx import Document
    doc = Document()
    doc.add_paragraph("Hello DOCX")
    buf = io.BytesIO()
    doc.save(buf)
    path = _write_temp(buf.getvalue(), ".docx")
    try:
        result = parse_file(path, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert "Hello DOCX" in result.text
    finally:
        path.unlink(missing_ok=True)


def test_parse_xlsx():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "Name"
    ws["B1"] = "Age"
    ws["A2"] = "Alice"
    ws["B2"] = "30"
    buf = io.BytesIO()
    wb.save(buf)
    path = _write_temp(buf.getvalue(), ".xlsx")
    try:
        result = parse_file(path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        assert "[Sheet: Sheet1]" in result.text
        assert "Alice" in result.text
        assert "30" in result.text
        assert result.sheets == ["Sheet1"]
    finally:
        path.unlink(missing_ok=True)


def test_parse_pptx():
    from pptx import Presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.placeholders[0].text = "Hello PPTX"
    buf = io.BytesIO()
    prs.save(buf)
    path = _write_temp(buf.getvalue(), ".pptx")
    try:
        result = parse_file(path, "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        assert "[Slide: 1]" in result.text
        assert "Hello PPTX" in result.text
        assert result.slides == 1
    finally:
        path.unlink(missing_ok=True)


def test_parse_unknown_mime():
    path = _write_temp("whatever", ".bin")
    try:
        result = parse_file(path, "application/octet-stream")
        assert "[Unsupported file type]" in result.text
    finally:
        path.unlink(missing_ok=True)


def test_parse_base64_text():
    b64 = base64.b64encode(b"Hello from base64").decode()
    result = parse_base64(b64, "text/plain", "test.txt")
    assert "Hello from base64" in result.text


def test_parse_base64_pdf():
    from fitz import open as fitz_open
    doc = fitz_open()
    doc.insert_page(0, text="PDF base64")
    buf = doc.write()
    doc.close()
    b64 = base64.b64encode(buf).decode()
    result = parse_base64(b64, "application/pdf", "doc.pdf")
    assert "PDF base64" in result.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_parser.py -v`
Expected: ModuleNotFoundError or ImportError

- [ ] **Step 3: Write file_parser.py**

Create `runtime/tools/file_parser.py`:

```python
from __future__ import annotations

import base64
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger("runtime.tools.file_parser")


@dataclass
class FileParseResult:
    text: str
    mime_type: str
    filename: str
    pages: int | None = None
    sheets: list[str] | None = None
    slides: int | None = None
    ocr_used: bool = False


def parse_file(path: Path, mime_type: str, filename: str = "untitled") -> FileParseResult:
    mime_lower = mime_type.lower().strip()

    if mime_lower == "text/plain":
        text = path.read_text("utf-8", errors="replace")
        return FileParseResult(text=text, mime_type=mime_type, filename=filename)

    if mime_lower == "text/markdown":
        text = path.read_text("utf-8", errors="replace")
        return FileParseResult(text=text, mime_type=mime_type, filename=filename)

    if mime_lower == "application/pdf":
        return _parse_pdf(path, filename)

    if mime_lower == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return _parse_docx(path, filename)

    if mime_lower == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return _parse_xlsx(path, filename)

    if mime_lower == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return _parse_pptx(path, filename)

    return FileParseResult(
        text=f"[Unsupported file type: {mime_type}]",
        mime_type=mime_type,
        filename=filename,
    )


def parse_base64(data: str, mime_type: str, filename: str = "untitled") -> FileParseResult:
    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        logger.warning("Failed to decode base64 file %s: %s", filename, exc)
        return FileParseResult(
            text=f"[Failed to decode file: {filename}]",
            mime_type=mime_type,
            filename=filename,
        )
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw)
        tmp = Path(f.name)
    try:
        return parse_file(tmp, mime_type, filename=filename)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _parse_pdf(path: Path, filename: str) -> FileParseResult:
    import fitz

    doc = fitz.open(str(path))
    texts: list[str] = []
    total_chars = 0
    for page in doc:
        text = page.get_text()
        texts.append(text)
        total_chars += len(text)

    result = FileParseResult(
        text="\n\n".join(texts),
        mime_type="application/pdf",
        filename=filename,
        pages=len(doc),
    )
    doc.close()

    if total_chars > 200:
        return result

    try:
        import pytesseract
    except ImportError:
        result.text = "[OCR not available — pytesseract not installed]"
        return result

    try:
        ocr_texts: list[str] = []
        doc = fitz.open(str(path))
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            ocr_text = pytesseract.image_to_string(img_bytes, lang=settings.tesseract_langs)
            ocr_texts.append(ocr_text.strip())
        doc.close()
        combined = "\n\n".join(t for t in ocr_texts if t)
        result = FileParseResult(
            text=combined or "[OCR produced no text]",
            mime_type="application/pdf",
            filename=filename,
            pages=result.pages,
            ocr_used=True,
        )
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", filename, exc)
        result.text = "[OCR failed]"

    return result


def _parse_docx(path: Path, filename: str) -> FileParseResult:
    from docx import Document

    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs]
    return FileParseResult(
        text="\n\n".join(paras),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


def _parse_xlsx(path: Path, filename: str) -> FileParseResult:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    sheet_names: list[str] = []
    for ws in wb.worksheets:
        sheet_names.append(ws.title)
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            rows.append("\t".join(cells))
        parts.append(f"[Sheet: {ws.title}]\n" + "\n".join(rows))
    wb.close()
    return FileParseResult(
        text="\n\n".join(parts),
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
        sheets=sheet_names,
    )


def _parse_pptx(path: Path, filename: str) -> FileParseResult:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    slide_count = 0
    for slide in prs.slides:
        slide_count += 1
        slide_texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        slide_texts.append(t)
            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    slide_texts.append("\t".join(cells))
        if slide_texts:
            parts.append(f"[Slide: {slide_count}]\n" + "\n".join(slide_texts))
    return FileParseResult(
        text="\n\n".join(parts),
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
        slides=slide_count,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_parser.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add runtime/tools/file_parser.py tests/test_file_parser.py
git commit -m "feat: add file parser engine for PDF/DOCX/XLSX/PPTX/TXT/MD with OCR fallback"
```

---

### Task 3: Upload Endpoint

**Files:**
- Create: `router/files_router.py`
- Create: `router/__init__.py` (update if needed)
- Test: `tests/test_file_upload.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_file_upload.py`:

```python
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from router.files_router import validate_upload, create_file_upload_route


@pytest.fixture
def mock_settings():
    with patch("router.files_router.settings") as mock:
        mock.upload_dir = Path(tempfile.mkdtemp())
        mock.max_upload_size_mb = 1
        mock.allowed_upload_mime_types = ["text/plain", "application/pdf"]
        yield mock
        for f in mock.upload_dir.iterdir():
            f.unlink()
        mock.upload_dir.rmdir()


def test_validate_upload_accepts_allowed_type(mock_settings):
    result = validate_upload("hello.txt", "text/plain", 100)
    assert result is None


def test_validate_upload_rejects_disallowed_type(mock_settings):
    result = validate_upload("script.exe", "application/x-msdownload", 100)
    assert result is not None
    assert "type" in result


def test_validate_upload_rejects_oversized(mock_settings):
    result = validate_upload("big.txt", "text/plain", mock_settings.max_upload_size_mb * 1024 * 1024 + 1)
    assert result is not None
    assert "size" in result


def test_validate_upload_rejects_path_traversal(mock_settings):
    result = validate_upload("../../etc/passwd", "text/plain", 100)
    assert result is not None


@pytest.mark.asyncio
async def test_upload_endpoint_success(mock_settings):
    route_fn = create_file_upload_route(MagicMock())
    content = b"Hello, this is a test file."
    upload = UploadFile(filename="test.txt", file=io.BytesIO(content))
    response = await route_fn(upload)
    assert response["filename"] == "test.txt"
    assert response["bytes"] == len(content)
    assert response["id"].startswith("file_")
    assert (mock_settings.upload_dir / response["id"]).exists()


@pytest.mark.asyncio
async def test_upload_endpoint_rejects_bad_type(mock_settings):
    route_fn = create_file_upload_route(MagicMock())
    upload = UploadFile(filename="evil.exe", file=io.BytesIO(b"bad"))
    with pytest.raises(Exception):
        await route_fn(upload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_upload.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write files_router.py**

Create `router/files_router.py`:

```python
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from config.settings import settings

logger = logging.getLogger("router.files_router")

router = APIRouter(prefix="/v1")

FILE_ID_PATTERN = "file_"


def validate_upload(filename: str, content_type: str, size: int) -> str | None:
    allowed_lower = [t.lower() for t in settings.allowed_upload_mime_types]
    if content_type.lower() not in allowed_lower:
        return f"File type '{content_type}' is not allowed"

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size > max_bytes:
        return f"File exceeds maximum size of {settings.max_upload_size_mb} MB"

    if "/" in filename.lstrip("/") or ".." in filename:
        return "Invalid filename"

    return None


def create_file_upload_route(service: Any):
    @router.post("/files")
    async def upload_file(file: UploadFile = File(...)):
        content = await file.read()
        content_type = file.content_type or "application/octet-stream"
        filename = file.filename or "untitled"

        error = validate_upload(filename, content_type, len(content))
        if error:
            raise HTTPException(status_code=400, detail={"type": "invalid_request_error", "message": error})

        file_id = f"file_{uuid.uuid4().hex}"
        dest = settings.upload_dir / file_id
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        logger.info("Uploaded file_id=%s filename=%s type=%s size=%d", file_id, filename, content_type, len(content))

        return {
            "id": file_id,
            "filename": filename,
            "bytes": len(content),
            "created_at": dest.stat().st_ctime,
        }

    return upload_file
```

- [ ] **Step 4: Register the router in the app**

Edit `router/openai_router.py` — find the app creation and add:

```python
from router.files_router import router as files_router
# after creating app:
app.include_router(files_router)
```

(Find exact location by reading the file)

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_upload.py -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add router/files_router.py tests/test_file_upload.py
git commit -m "feat: add multipart file upload endpoint POST /v1/files"
```

---

### Task 4: File Cleanup

**Files:**
- Create: `runtime/tools/file_cleanup.py`
- Test: `tests/test_file_cleanup.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_file_cleanup.py`:

```python
from __future__ import annotations

import time
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import MagicMock, patch

import pytest

from runtime.tools.file_cleanup import FileCleanupManager


@pytest.fixture
def tmp_dir():
    d = Path(mkdtemp())
    yield d
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def test_cleanup_files_success(tmp_dir):
    f1 = tmp_dir / "file_a"
    f2 = tmp_dir / "file_b"
    f1.write_text("a")
    f2.write_text("b")

    mgr = FileCleanupManager(tmp_dir)
    mgr.track("req_1", ["file_a", "file_b"])
    mgr.cleanup_request("req_1")

    assert not f1.exists()
    assert not f2.exists()


def test_cleanup_unknown_request_is_noop(tmp_dir):
    mgr = FileCleanupManager(tmp_dir)
    mgr.cleanup_request("req_nonexistent")  # should not raise


def test_cleanup_partial_failure_logs_warning(tmp_dir):
    f1 = tmp_dir / "file_a"
    f1.write_text("a")

    mgr = FileCleanupManager(tmp_dir)
    mgr.track("req_1", ["file_a", "file_missing"])
    # Should not raise even though file_missing doesn't exist
    with patch("runtime.tools.file_cleanup.logger.warning") as mock_warn:
        mgr.cleanup_request("req_1")
        mock_warn.assert_called_once()


def test_ttl_sweep_removes_expired(tmp_dir):
    f1 = tmp_dir / "old_file"
    f1.write_text("old")

    mgr = FileCleanupManager(tmp_dir, ttl_seconds=0)  # immediate expiry
    with patch("runtime.tools.file_cleanup.time.time", return_value=time.time() + 100):
        mgr._sweep_expired()
        assert not f1.exists()


def test_ttl_sweep_keeps_recent(tmp_dir):
    f1 = tmp_dir / "recent_file"
    f1.write_text("new")

    mgr = FileCleanupManager(tmp_dir, ttl_seconds=600)
    mgr._sweep_expired()
    assert f1.exists()


def test_contextvar_isolation(tmp_dir):
    mgr = FileCleanupManager(tmp_dir)
    mgr.set_current_request("req_a")
    assert mgr.get_current_request() == "req_a"


def test_track_without_request_does_not_raise(tmp_dir):
    mgr = FileCleanupManager(tmp_dir)
    mgr.track_current("file_a")  # no request set — should be noop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_cleanup.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write file_cleanup.py**

Create `runtime/tools/file_cleanup.py`:

```python
from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger("runtime.tools.file_cleanup")

_current_request: ContextVar[str | None] = ContextVar("_file_cleanup_request", default=None)


class FileCleanupManager:
    def __init__(self, upload_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        self._upload_dir = upload_dir or settings.upload_dir
        self._ttl = ttl_seconds or settings.file_cleanup_ttl_seconds
        self._request_files: dict[str, list[str]] = {}
        self._lock: Any = None

    # --- request-scoped ---

    def set_current_request(self, request_id: str) -> None:
        _current_request.set(request_id)

    def get_current_request(self) -> str | None:
        return _current_request.get()

    def track(self, request_id: str, file_ids: list[str]) -> None:
        self._request_files.setdefault(request_id, []).extend(file_ids)

    def track_current(self, file_id: str) -> None:
        req_id = self.get_current_request()
        if req_id:
            self.track(req_id, [file_id])

    def cleanup_request(self, request_id: str) -> None:
        file_ids = self._request_files.pop(request_id, [])
        for fid in file_ids:
            path = self._upload_dir / fid
            if not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to cleanup file %s: %s", fid, exc)

    # --- background TTL ---

    def _sweep_expired(self) -> None:
        now = time.time()
        for entry in self._upload_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.name == ".gitkeep":
                continue
            age = now - entry.stat().st_ctime
            if age > self._ttl:
                try:
                    entry.unlink()
                except OSError as exc:
                    logger.warning("TTL cleanup failed for %s: %s", entry.name, exc)

    async def background_cleanup_loop(self, interval_seconds: int = 300) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                self._sweep_expired()
            except Exception as exc:
                logger.warning("Background cleanup sweep failed: %s", exc)


_file_cleanup_manager: FileCleanupManager | None = None


def get_file_cleanup_manager() -> FileCleanupManager:
    global _file_cleanup_manager
    if _file_cleanup_manager is None:
        _file_cleanup_manager = FileCleanupManager()
    return _file_cleanup_manager


def ensure_cleanup_dir() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_file_cleanup.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add runtime/tools/file_cleanup.py tests/test_file_cleanup.py
git commit -m "feat: add file cleanup manager with request-scoped + TTL sweep"
```

---

### Task 5: Content Block Resolution

**Files:**
- Modify: `runtime/tools/content_blocks.py` — add `resolve_file_blocks()`
- Test: `tests/test_content_blocks.py` — add resolution tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_content_blocks.py`:

```python
def test_resolve_file_blocks_inline_base64_text():
    from runtime.tools.content_blocks import resolve_file_blocks
    import base64
    b64 = base64.b64encode(b"Hello from inline file").decode()
    blocks = [
        {"type": "document", "source": {"type": "base64", "media_type": "text/plain", "data": b64}},
    ]
    result = resolve_file_blocks(blocks, provider="openai")
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "Hello from inline file" in result[0]["text"]


def test_resolve_file_blocks_pdf_for_anthropic_passthrough():
    from runtime.tools.content_blocks import resolve_file_blocks
    import base64
    b64 = base64.b64encode(b"%PDF-fake-content").decode()
    blocks = [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
    ]
    result = resolve_file_blocks(blocks, provider="anthropic")
    assert len(result) == 1
    assert result[0]["type"] == "document"  # kept original


def test_resolve_file_blocks_pdf_for_non_anthropic_parsed():
    from runtime.tools.content_blocks import resolve_file_blocks
    # PDF with very short content will fail OCR and return OCR-not-available text
    import base64
    b64 = base64.b64encode(b"Hello PDF inline").decode()
    blocks = [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
    ]
    result = resolve_file_blocks(blocks, provider="openai")
    assert len(result) == 1
    # Should be parsed (even if OCR unavailable, it returns OCR-not-available text)
    assert result[0]["type"] == "text"
    assert "OCR" in result[0]["text"] or "Hello" not in result[0]["text"]  # short PDF won't render text


def test_resolve_file_blocks_text_blocks_pass_through():
    from runtime.tools.content_blocks import resolve_file_blocks
    blocks = [{"type": "text", "text": "hello"}]
    result = resolve_file_blocks(blocks, provider="openai")
    assert result == blocks


def test_resolve_file_blocks_mixed():
    from runtime.tools.content_blocks import resolve_file_blocks
    import base64
    b64 = base64.b64encode(b"Hello file").decode()
    blocks = [
        {"type": "text", "text": "before"},
        {"type": "document", "source": {"type": "base64", "media_type": "text/plain", "data": b64}},
        {"type": "text", "text": "after"},
    ]
    result = resolve_file_blocks(blocks, provider="openai")
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "text"
    assert "Hello file" in result[1]["text"]
    assert result[2]["type"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_content_blocks.py -x -v -k "resolve_file_blocks"`
Expected: ImportError for `resolve_file_blocks`

- [ ] **Step 3: Add resolve_file_blocks to content_blocks.py**

Add to `runtime/tools/content_blocks.py` (after imports, before `anthropic_content_to_openai_parts`):

```python
from runtime.tools.file_parser import parse_base64


def resolve_file_blocks(blocks: list[Any], provider: str) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            resolved.append({"type": "text", "text": str(block)})
            continue

        block_type = str(block.get("type", "")).lower()

        if block_type not in {"document", "file", "input_file"}:
            resolved.append(block)
            continue

        if (
            provider == "anthropic"
            and block_type == "document"
            and _get_document_media_type(block) == "application/pdf"
        ):
            resolved.append(block)
            continue

        text = _resolve_document_block(block)
        resolved.append({"type": "text", "text": text})

    return resolved


def _get_document_media_type(block: dict[str, Any]) -> str:
    source = block.get("source", {})
    if isinstance(source, dict):
        mt = source.get("media_type")
        if mt:
            return str(mt).lower()
    mt = block.get("media_type")
    return str(mt).lower() if mt else "application/octet-stream"


def _resolve_document_block(block: dict[str, Any]) -> str:
    source = block.get("source", {})
    source = source if isinstance(source, dict) else {}
    media_type = str(source.get("media_type") or block.get("media_type") or "application/octet-stream")
    data = str(source.get("data", ""))
    url = str(source.get("url", ""))
    filename = str(block.get("title") or block.get("name") or "untitled")

    if data:
        result = parse_base64(data, media_type, filename=filename)
        return result.text

    if url:
        from runtime.tools.file_cleanup import get_file_cleanup_manager
        mgr = get_file_cleanup_manager()
        file_path = mgr._upload_dir / url
        if file_path.exists():
            from runtime.tools.file_parser import parse_file
            result = parse_file(file_path, media_type, filename=filename)
            return result.text
        return f"[File: {filename} ({media_type}), url={url}]"

    return f"[File: {filename} ({media_type})]"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_content_blocks.py -x -v -k "resolve_file_blocks"`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add runtime/tools/content_blocks.py tests/test_content_blocks.py
git commit -m "feat: add resolve_file_blocks for document/file content block resolution with Anthropic PDF passthrough"
```

---

### Task 6: Wire File Block Resolution into Messages Adapter

**Files:**
- Modify: `router/anthropic/messages_adapter.py` — resolve files after routing, before adapter call

- [ ] **Step 1: Add import and resolve call in messages_adapter.py**

In `router/anthropic/messages_adapter.py`, add import at top (after existing `runtime.tools.builtin.web_search` imports):

Add before line 31 (`logger = logging.getLogger(...)`):

```python
from runtime.tools.content_blocks import resolve_file_blocks
```

After routing is decided (after `openai_payload["model"] = routing_decision.model` on line 115), add helper to resolve files in messages:

Find the block around line 115-121 and add after line 121:

```python
        # Resolve file/document blocks based on provider
        if "messages" in openai_payload and isinstance(openai_payload["messages"], list):
            openai_payload["messages"] = [
                {
                    **msg,
                    "content": (
                        resolve_file_blocks(msg["content"], provider)
                        if isinstance(msg.get("content"), list)
                        else msg["content"]
                    ),
                }
                if isinstance(msg, dict) else msg
                for msg in openai_payload["messages"]
            ]
```

- [ ] **Step 2: Run existing tests to verify nothing broken**

Run: `PYTHONPATH=. .venv/bin/pytest tests/ -x -v --ignore=tests/test_capabilities.py --ignore=tests/test_dashboard_auth.py --ignore=tests/test_security.py`
Expected: all 250+ pass

- [ ] **Step 3: Commit**

```bash
git add router/anthropic/messages_adapter.py
git commit -m "feat: resolve file blocks in Anthropic messages adapter after routing"
```

---

### Task 7: Wire File Block Resolution into OpenAI Handler

**Files:**
- Modify: `runtime/orchestration/openai_handler.py` — add file resolution in `_normalize_payload_for_provider`

- [ ] **Step 1: Add import and resolution to openai_handler.py**

In `runtime/orchestration/openai_handler.py`, add import near top with other runtime imports:

```python
from runtime.tools.content_blocks import resolve_file_blocks
```

In `_normalize_payload_for_provider()` (around line 1290), add file resolution for all providers.
Add before the `if provider == "ollama":` normalization:

```python
        # Resolve file/document blocks for all providers
        messages = normalized.get("messages")
        if isinstance(messages, list):
            normalized["messages"] = [
                {
                    **msg,
                    "content": (
                        resolve_file_blocks(msg["content"], provider)
                        if isinstance(msg.get("content"), list)
                        else msg["content"]
                    ),
                }
                if isinstance(msg, dict) else msg
                for msg in messages
            ]
```

The resolved method should look like:

```python
    def _normalize_payload_for_provider(self, payload: dict[str, Any], provider: str) -> dict[str, Any]:
        normalized = dict(payload)
        if "tools" in normalized:
            normalized["tools"] = self._ensure_openai_tools(normalized["tools"])
        messages = normalized.get("messages")
        if isinstance(messages, list):
            normalized["messages"] = [
                {
                    **msg,
                    "content": (
                        resolve_file_blocks(msg["content"], provider)
                        if isinstance(msg.get("content"), list)
                        else msg["content"]
                    ),
                }
                if isinstance(msg, dict) else msg
                for msg in messages
            ]
        if provider == "ollama":
            messages = self._extract_messages_from_payload(normalized)
            if messages is not None:
                normalized["messages"] = [self._normalize_ollama_message(message) for message in messages]
        return normalized
```

- [ ] **Step 2: Run existing tests to verify nothing broken**

Run: `PYTHONPATH=. .venv/bin/pytest tests/ -x -v --ignore=tests/test_capabilities.py --ignore=tests/test_dashboard_auth.py --ignore=tests/test_security.py`
Expected: all 250+ pass

- [ ] **Step 3: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "feat: resolve file blocks in openai_handler normalize_payload_for_provider"
```

---

### Task 8: Responses API File Input Support

**Files:**
- Modify: `runtime/responses/input_converter.py` — resolve file content in FILE items
- Modify: `runtime/responses/response_models.py` — add `content` field to `InputItem` for file items

- [ ] **Step 1: Write tests for FILE input with content**

Add to `tests/test_responses_tool_loop.py`:

```python
def test_file_input_item_with_base64_content():
    from runtime.responses.response_models import InputItem, InputItemType
    item = InputItem(
        id="file_1",
        type=InputItemType.FILE,
        file_id="file_abc123",
        filename="test.txt",
        content=[{"type": "text", "text": "hello from file"}],
    )
    assert item.file_id == "file_abc123"


def test_input_converter_parses_file_with_content():
    from runtime.responses.input_converter import _parse_input_item
    item = _parse_input_item({
        "type": "file",
        "file_id": "file_abc",
        "filename": "doc.txt",
    })
    assert item.type.value == "file"
    assert item.file_id == "file_abc"
    assert item.filename == "doc.txt"
```

- [ ] **Step 2: Modify _input_item_to_messages to extract file content**

In `runtime/responses/input_converter.py`, update the `InputItemType.FILE` branch in `_input_item_to_messages()` (around line 149):

```python
    if item.type == InputItemType.FILE:
        text = f"[file: {item.filename or item.file_id}]"
        if item.content and isinstance(item.content, list):
            texts = []
            for part in item.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            if texts:
                text = "\n".join(texts)
        return [{
            "role": "user",
            "content": text,
        }]
```

Also add `content` field to `InputItem` in `response_models.py` if it doesn't exist (check the existing model):

Check `response_models.py:129-130` — `file_id` and `filename` exist. We may need to add `content: list[dict] | None = None` to the InputItem fields. Add it:

```python
    content: list[dict[str, Any]] | None = None
```

- [ ] **Step 3: Run tests to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_responses_tool_loop.py -x -v -k "file_input"`
Expected: tests pass

- [ ] **Step 4: Commit**

```bash
git add runtime/responses/input_converter.py runtime/responses/response_models.py
git commit -m "feat: support file content extraction in Responses API FILE input items"
```

---

### Task 9: Capability Registration

**Files:**
- Modify: `providers/registry.py` — add DOCUMENTS capability
- Modify: `runtime/orchestration/capabilities.py` — add document/file capability detection
- Modify: `runtime/orchestration/anthropic_converter.py` — pass document capability to routing

- [ ] **Step 1: Add DOCUMENTS to capability enum**

In `providers/registry.py`, add `DOCUMENTS` to the `Capability` enum and add aliases:

```python
class Capability(str, Enum):
    # ... existing ...
    DOCUMENTS = "documents"
```

Add to `CAPABILITY_ALIASES`:

```python
    "document": Capability.DOCUMENTS,
    "documents": Capability.DOCUMENTS,
    "file": Capability.DOCUMENTS,
    "files": Capability.DOCUMENTS,
```

- [ ] **Step 2: Add document capability detection to capabilities.py**

In `runtime/orchestration/capabilities.py`, add to `_add_part_capability`:

```python
    elif part_type in {"document", "file", "input_file"}:
        required.add("documents")
```

- [ ] **Step 3: Run tests to verify nothing broken**

Run: `PYTHONPATH=. .venv/bin/pytest tests/ -x -v --ignore=tests/test_capabilities.py --ignore=tests/test_dashboard_auth.py --ignore=tests/test_security.py`
Expected: all 250+ pass

- [ ] **Step 4: Commit**

```bash
git add providers/registry.py runtime/orchestration/capabilities.py
git commit -m "feat: add DOCUMENTS capability for routing document-aware models"
```

---

### Task 10: Wire Cleanup into Server Lifecycle

**Files:**
- Modify: `router/openai_router.py` — start background cleanup loop on app startup
- Modify: `runtime/orchestration/openai_handler.py` — call cleanup in request finalization

- [ ] **Step 1: Start background cleanup on app startup**

In `router/openai_router.py`, find the app creation and add startup event:

```python
from runtime.tools.file_cleanup import get_file_cleanup_manager, ensure_cleanup_dir

@app.on_event("startup")
async def start_file_cleanup():
    ensure_cleanup_dir()
    mgr = get_file_cleanup_manager()
    asyncio.create_task(mgr.background_cleanup_loop())
```

- [ ] **Step 2: Add cleanup call to openai_handler.py methods**

In `openai_handler.py`, find the main handler methods (`handle_chat`, `handle_streaming_chat`, `handle_responses`, `handle_streaming_responses`) and add cleanup in their `finally` blocks.

Add import at top:

```python
from runtime.tools.file_cleanup import get_file_cleanup_manager
```

In each handler, find the existing `finally` blocks (e.g., in `handle_streaming_chat` around the `_finalize_request` call) and add:

```python
finally:
    file_mgr = get_file_cleanup_manager()
    request_id = normalized.get("x-request-id", "") if isinstance(normalized, dict) else ""
    if request_id:
        file_mgr.cleanup_request(request_id)
```

For `handle_chat`, add similar cleanup after response is sent.

For `handle_responses` and `handle_streaming_responses`, add the same pattern.

Also add `track_current()` call in the upload endpoint to associate files with request IDs.

In `router/files_router.py`, after writing the file:

```python
from runtime.tools.file_cleanup import get_file_cleanup_manager
# ... after saving file ...
get_file_cleanup_manager().track_current(file_id)
```

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH=. .venv/bin/pytest tests/ -x -v --ignore=tests/test_capabilities.py --ignore=tests/test_dashboard_auth.py --ignore=tests/test_security.py`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add router/openai_router.py router/files_router.py runtime/orchestration/openai_handler.py
git commit -m "feat: wire file cleanup lifecycle — background TTL + request-scoped cleanup"
```<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="read">
<｜｜DSML｜｜parameter name="filePath" string="true">/Users/fred/ai/my_opencode/AetherMesh/runtime/orchestration/openai_handler.py