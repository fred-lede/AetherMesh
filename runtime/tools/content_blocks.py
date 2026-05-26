from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger("runtime.tools.content_blocks")


def resolve_file_blocks(
    file_ids: list[str],
    provider: str,
    upload_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if upload_dir is None:
        upload_dir = settings.upload_dir

    blocks: list[dict[str, Any]] = []
    provider_lower = provider.strip().lower()

    for file_id in file_ids:
        file_path = upload_dir / file_id
        meta_path = upload_dir / f"{file_id}.meta.json"

        if not file_path.exists() or not meta_path.exists():
            logger.warning("File not found for resolve: %s", file_id)
            continue

        meta = json.loads(meta_path.read_text("utf-8"))
        mime_type = meta.get("mime_type", "application/octet-stream")

        if provider_lower == "anthropic" and mime_type == "application/pdf":
            data_bytes = file_path.read_bytes()
            b64_data = base64.b64encode(data_bytes).decode("ascii")
            blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64_data,
                },
            })
        else:
            content_text = file_path.read_text("utf-8", errors="replace")
            blocks.append({
                "type": "text",
                "text": content_text,
            })

    return blocks


def anthropic_content_to_openai_parts(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        parts.extend(anthropic_block_to_openai_parts(block))
    return parts


def anthropic_block_to_openai_parts(block: Any) -> list[dict[str, Any]]:
    if isinstance(block, str):
        return [{"type": "text", "text": block}]
    if not isinstance(block, dict):
        return []

    block_type = str(block.get("type", "")).lower()

    if block_type == "text":
        return [_with_cache({"type": "text", "text": str(block.get("text", ""))}, block)]

    if block_type == "image":
        image = _anthropic_image_to_openai(block)
        return [_with_cache(image, block)] if image else []

    if block_type == "document":
        return [_with_cache({"type": "text", "text": _document_hint_or_text(block)}, block)]

    if block_type == "tool_result":
        return _tool_result_to_openai_parts(block)

    if block_type == "thinking":
        thinking_text = str(block.get("thinking", ""))
        return [{"type": "text", "text": f"[thinking] {thinking_text} [/thinking]"}]

    if block_type in {"audio", "input_audio"}:
        audio = _anthropic_audio_to_openai(block)
        if audio:
            return [_with_cache(audio, block)]
        return [_with_cache({"type": "text", "text": _audio_hint(block)}, block)]

    return [{"type": "text", "text": str(block)}]


def content_part_to_text_and_images(part: Any) -> tuple[str, list[str]]:
    if isinstance(part, str):
        return part, []
    if not isinstance(part, dict):
        return "", []

    part_type = str(part.get("type", "")).lower()
    if part_type in {"text", "input_text", "output_text"}:
        return str(part.get("text", "")), []

    if "text" in part and isinstance(part.get("text"), str):
        return str(part.get("text", "")), []

    if part_type in {"image_url", "input_image", "image"}:
        image = image_ref_from_content_part(part)
        return "", [image] if image else []

    if part_type in {"document", "input_file", "file"}:
        return _openai_document_hint(part), []

    if part_type in {"input_audio", "audio"}:
        return _openai_audio_hint(part), []

    return "", []


def image_ref_from_content_part(part: dict[str, Any]) -> str:
    image_value = part.get("image_url")
    if isinstance(image_value, dict):
        url = str(image_value.get("url", ""))
    elif image_value is None and isinstance(part.get("url"), str):
        url = str(part.get("url", ""))
    else:
        url = str(image_value or "")

    if not url and isinstance(part.get("image"), str):
        url = str(part.get("image", ""))

    return normalize_image_ref(url)


def normalize_image_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("data:") and "," in raw:
        _, data = raw.split(",", 1)
        return data.strip()
    return raw


def _with_cache(part: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    cache = source.get("cache_control")
    if cache:
        part = dict(part)
        part["cache_control"] = cache
    return part


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source", {})
    if not isinstance(source, dict):
        return None

    media_type = source.get("media_type", "image/jpeg")
    data = str(source.get("data", ""))
    url = str(source.get("url", ""))

    if data:
        if not data.startswith("data:"):
            data = f"data:{media_type};base64,{data}"
        return {
            "type": "image_url",
            "image_url": {"url": data, "detail": block.get("detail", "auto")},
        }
    if url:
        return {
            "type": "image_url",
            "image_url": {"url": url, "detail": block.get("detail", "auto")},
        }
    return None


def _document_hint_or_text(block: dict[str, Any]) -> str:
    source = block.get("source", {})
    source = source if isinstance(source, dict) else {}
    media_type = str(
        source.get("media_type") or block.get("media_type") or "application/octet-stream"
    )
    title = str(block.get("title") or block.get("name") or "untitled")
    data = str(source.get("data", ""))
    url = str(source.get("url", ""))

    if media_type == "text/plain" and data:
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            return f"[Document: {title}]\n{decoded}" if title else decoded
        except Exception:
            return f"[Document: {title} ({media_type}) - failed to decode]"

    if url:
        return f"[Document: {title} ({media_type}), url={url}]"
    if data:
        return f"[Document: {title} ({media_type}), base64 encoded]"
    return f"[Document: {title} ({media_type})]"


def _tool_result_to_openai_parts(block: dict[str, Any]) -> list[dict[str, Any]]:
    tool_use_id = str(block.get("tool_use_id", ""))
    content_val = block.get("content", "")
    is_error = bool(block.get("is_error", False))
    prefix = "[Tool Error" if is_error else "[Tool Result"
    label = f"{prefix} ({tool_use_id})]: " if tool_use_id else f"{prefix}]: "

    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    if isinstance(content_val, list):
        for item in content_val:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") == "image":
                image = _anthropic_image_to_openai(item)
                if image:
                    image_parts.append(image)
                else:
                    text_parts.append("[image]")
            elif isinstance(item, dict) and item.get("type") == "document":
                text_parts.append(_document_hint_or_text(item))
            elif isinstance(item, dict) and item.get("type") in {"audio", "input_audio"}:
                text_parts.append(_audio_hint(item))
            elif isinstance(item, dict):
                text_parts.append(str(item))
    else:
        text_parts.append(str(content_val))

    text = label + "\n".join(part for part in text_parts if part)
    result: list[dict[str, Any]] = []
    if text.strip():
        result.append(_with_cache({"type": "text", "text": text}, block))
    result.extend(_with_cache(part, block) for part in image_parts)
    return result


def _anthropic_audio_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source", {})
    source = source if isinstance(source, dict) else {}
    data = source.get("data") or block.get("data")
    if not data:
        return None

    media_type = str(source.get("media_type") or block.get("media_type") or "audio/wav")
    audio_format = media_type.rsplit("/", 1)[-1] if "/" in media_type else media_type
    return {
        "type": "input_audio",
        "input_audio": {
            "data": str(data),
            "format": audio_format,
        },
    }


def _audio_hint(block: dict[str, Any]) -> str:
    source = block.get("source", {})
    source = source if isinstance(source, dict) else {}
    media_type = str(source.get("media_type") or block.get("media_type") or "audio")
    url = source.get("url") or block.get("url")
    if url:
        return f"[Audio: {media_type}, url={url}]"
    return f"[Audio: {media_type}]"


def _openai_document_hint(part: dict[str, Any]) -> str:
    name = part.get("filename") or part.get("name") or part.get("file_id") or "untitled"
    return f"[Document: {name}]"


def _openai_audio_hint(part: dict[str, Any]) -> str:
    audio = part.get("input_audio")
    if isinstance(audio, dict):
        fmt = audio.get("format") or "audio"
        return f"[Audio: {fmt}]"
    return "[Audio]"
