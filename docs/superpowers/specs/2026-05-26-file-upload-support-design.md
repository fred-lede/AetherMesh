# File Upload & Document Support — Design

**Date**: 2026-05-26
**Status**: Draft
**Owner**: Agent

## Problem

AetherMesh has no file upload capability. All APIs accept JSON-only bodies. PDF and Office file content cannot be sent to LLM providers — PDFs passed as Anthropic `document` blocks are replaced with `[Document: filename (type)]` text hints, and Office formats are not supported at all.

## Scope

Support **PDF, DOCX, XLSX, PPTX, TXT, MD** files uploaded via:
1. Multipart upload API (`POST /v1/files`) → returns `file_id` for later reference
2. Inline base64 (existing Anthropic `document` block format)

## Architecture

```
Client                    AetherMesh
  │                          │
  ├─ POST /v1/files ────────→│  router/files_router.py
  │   (multipart)            │    → save to upload_dir (/tmp/aethermesh/uploads/)
  │   ← {file_id, filename}  │    → return file_id
  │                          │
  ├─ POST /v1/chat/completions ──→ normal routing
  │   { messages: [{ content:  │
  │     [{type:"file",         │
  │       file_id:"file_abc"}] │
  │     }] }                   │
  │                          │
  │                          ├─ resolve_file_blocks() in content_blocks.py
  │                          │    → read file from upload_dir
  │                          │    → parse via file_parser.py
  │                          │    → inject text block, remove file/document block
  │                          │
  │                          ├─ provider-specific:
  │                          │    Anthropic + PDF native → keep original document block
  │                          │    Others → use parsed text
  │                          │
  │                          └─ finally: cleanup_temp_files(request_id)
```

## Components

### 1. `router/files_router.py` — Upload Endpoint

```python
@router.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
```

- MIME type validation against `settings.allowed_upload_mime_types`
- Size limit via `settings.max_upload_size_mb` (default 50 MB)
- Store to `settings.upload_dir / file_id`
- Return `{id, filename, bytes, created_at}`

### 2. `runtime/tools/file_parser.py` — Parse Engine

```python
@dataclass
class FileParseResult:
    text: str
    mime_type: str
    filename: str
    pages: int | None = None
    sheets: list[str] | None = None
    slides: int | None = None
    ocr_used: bool = False

def parse_file(path: Path, mime_type: str) -> FileParseResult:
```

| Format | Library | Strategy |
|---|---|---|
| **PDF** | PyMuPDF (`fitz`) | `page.get_text()` → if >200 chars total → text OK. Else render + Tesseract OCR → text |
| **DOCX** | `python-docx` | paragraphs.text joined with `\n\n` |
| **XLSX** | `openpyxl` | Per-sheet: tab-separated rows, prefixed `[Sheet: name]` |
| **PPTX** | `python-pptx` | Per-slide: text frames + tables, prefixed `[Slide: N]` |
| **TXT/MD** | native | Read UTF-8 directly |

### 3. `runtime/tools/content_blocks.py` — File Block Resolution

- New function `resolve_file_blocks(content_blocks, provider)`:
  - Iterates content blocks
  - For `file`/`input_file`/`document` blocks: read file, parse, convert to text
  - If provider == `"anthropic"` and block is PDF document → keep original block
  - Otherwise → replace with `{"type": "text", "text": parsed_text}`
- Called before `anthropic_content_to_openai_parts()` for non-Anthropic providers
- Called in `anthropic_converter.py::_to_openai_payload()` for Anthropic (to keep pass-through)

### 4. `runtime/tools/file_cleanup.py` — Cleanup

Two layers:

**Level 1 — Request-scoped** (`finally` in `handle_chat`/`handle_streaming_chat`):
- Track `file_ids` per request via contextvar
- After response sent → delete files from disk
- Log warning on failure (don't crash)

**Level 2 — Background TTL** (asyncio task, starts with server):
```python
async def _file_cleanup_loop():
    while True:
        await asyncio.sleep(300)  # 5 min
        for f in upload_dir.iterdir():
            if f.stat().st_ctime + 600 < time.time():
                f.unlink(missing_ok=True)
```

### 5. Provider Integration Points

| Point | File | What changes |
|---|---|---|
| Content block resolution | `content_blocks.py` | New `resolve_file_blocks()` |
| Anthropic converter | `anthropic_converter.py` | Call resolve before convert, keep PDF blocks |
| OpenAI chat adapter | `chat_adapter.py` | Call resolve before forwarding |
| Responses input converter | `input_converter.py` | Parse `file_id` in `InputItemType.FILE` |
| Capabilities | `registry.py` | New `DOCUMENTS` capability |
| Settings | `settings.py` | New config fields |

### 6. Capability Registration

```python
class Capability(StrEnum):
    DOCUMENTS = "documents"
```

Models that support document native handling (e.g. Anthropic claude-3-5-sonnet for PDF) register `DOCUMENTS`. If a model lacks this capability, all file content is resolved locally to text.

### 7. Security

- MIME type whitelist enforced at upload
- File size limit (50 MB default, configurable)
- Path traversal protection: `file_id` must match `^file_[a-f0-9]{32}$` regex
- OCR concurrency limited via `asyncio.Semaphore(2)`
- Upload dir created with `0o700` permissions

### 8. Settings Additions (`config/settings.py`)

| Field | Type | Default |
|---|---|---|
| `upload_dir` | `Path` | `/tmp/aethermesh/uploads/` |
| `max_upload_size_mb` | `int` | `50` |
| `allowed_upload_mime_types` | `list[str]` | see full list below |
| `tesseract_langs` | `str` | `"eng"` |
| `file_cleanup_ttl_seconds` | `int` | `600` |
| `ocr_concurrency` | `int` | `2` |

Allowed MIME types:
- `application/pdf`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (.docx)
- `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (.xlsx)
- `application/vnd.openxmlformats-officedocument.presentationml.presentation` (.pptx)
- `text/plain`, `text/markdown`

### 9. Test Plan

- `tests/test_file_parser.py`: parse each format (include small sample files)
- `tests/test_file_upload.py`: upload endpoint, validation, size/type rejection
- `tests/test_content_blocks.py`: file block resolution + provider passthrough
- `tests/test_file_cleanup.py`: success cleanup, failure resilience, TTL sweep
- Integration: upload → chat with file_id → assert text in response

## Non-Goals

- OCR for non-PDF images (JPEG/PNG → text) — not requested
- Persistent file storage (S3, DB) — start with temp disk only
- File versioning or status tracking
- Browser-based file picker UI
- Streaming file upload progress
