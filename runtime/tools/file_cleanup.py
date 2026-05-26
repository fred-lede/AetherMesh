from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from pathlib import Path

from config.settings import settings

logger = logging.getLogger("runtime.tools.file_cleanup")

_current_request: ContextVar[str | None] = ContextVar("_file_cleanup_request", default=None)


class FileCleanupManager:
    def __init__(self, upload_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        self._upload_dir = upload_dir or settings.upload_dir
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.file_cleanup_ttl_seconds
        self._request_files: dict[str, list[str]] = {}

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
                logger.warning("Tracked file not found: %s", fid)
                continue
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to cleanup file %s: %s", fid, exc)

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
