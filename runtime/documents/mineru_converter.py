from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger("documents.mineru_converter")


class MinerUError(RuntimeError):
    pass


def _mineru_command() -> list[str]:
    python_path = settings.mineru_python.strip()
    python: Path | None = None
    if python_path:
        candidate = Path(python_path)
        if candidate.exists():
            python = candidate
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        scripts = repo_root / ".venv312" / ("Scripts" if os.name == "nt" else "bin")
        candidate = scripts / ("python.exe" if os.name == "nt" else "python")
        if candidate.exists():
            python = candidate
    if python is not None:
        return [str(python), "-m", "mineru.cli.client"]
    which = shutil.which("mineru")
    if which:
        return [which]
    raise MinerUError(
        "MinerU CLI not found. Install into a separate venv (e.g. .venv312) "
        "and set AIIH_MINERU_PYTHON to its python executable."
    )


def mineru_available() -> bool:
    try:
        _mineru_command()
        return True
    except MinerUError:
        return False


def _find_markdown_output(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.rglob("*.md"))
    return candidates[0] if candidates else None


def _discover_images(out_dir: Path) -> list[dict[str, str]]:
    """Return list of {rel_path, abs_path, mime} for images under out_dir."""
    images: list[dict[str, str]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp"):
        for p in sorted(out_dir.rglob(ext)):
            rel = str(p.relative_to(out_dir))
            images.append({"rel_path": rel, "abs_path": str(p), "mime": _mime_for(p)})
    return images


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


def _embed_images_in_markdown(markdown: str, md_path: Path) -> str:
    """Replace ![](images/xxx.jpg) with base64 data URIs, resolving relative to the markdown file."""
    base_dir = md_path.parent

    def _replace(match: re.Match) -> str:
        rel = match.group(1)
        img_path = (base_dir / rel).resolve()
        if not img_path.exists():
            return match.group(0)
        try:
            data = img_path.read_bytes()
            mime = _mime_for(img_path)
            b64 = base64.b64encode(data).decode("ascii")
            return f"![](data:{mime};base64,{b64})"
        except OSError:
            return match.group(0)

    return re.sub(r"!\[\]\((images/[^)]+)\)", _replace, markdown)


def convert_document(
    file_path: str | Path,
    out_dir: str | Path | None = None,
    backend: str | None = None,
    method: str | None = None,
    timeout_s: int | None = None,
    include_images: bool = False,
) -> dict[str, Any]:
    source = Path(file_path)
    if not source.exists():
        raise MinerUError(f"File not found: {source}")
    out_dir = Path(out_dir or (source.parent / "mineru_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _mineru_command() + [
        "-p",
        str(source),
        "-o",
        str(out_dir),
        "-m",
        method or settings.mineru_method,
        "-b",
        backend or settings.mineru_backend,
    ]
    logger.info("Running MinerU: %s", " ".join(cmd))
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONEXECUTABLE", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME"):
        env.pop(key, None)
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s or settings.mineru_timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise MinerUError(f"MinerU timed out after {timeout_s or settings.mineru_timeout_s}s") from None
    duration_ms = int((time.time() - started) * 1000)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise MinerUError(f"MinerU exited with code {result.returncode}: {detail}")
    md_path = _find_markdown_output(out_dir)
    if md_path is None:
        raise MinerUError(f"MinerU finished but produced no markdown in {out_dir}")
    content = md_path.read_text(encoding="utf-8", errors="replace")
    images = _discover_images(out_dir)
    if include_images:
        content = _embed_images_in_markdown(content, md_path)
    return {
        "markdown": content,
        "source": str(source),
        "output_path": str(md_path),
        "duration_ms": duration_ms,
        "chars": len(content),
        "image_count": len(images),
        "images": [{"rel_path": i["rel_path"], "mime": i["mime"]} for i in images] if not include_images else [],
    }
