from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import mkdtemp

import pytest

from runtime.tools.file_cleanup import FileCleanupManager, ensure_cleanup_dir


def test_ensure_cleanup_dir(tmp_path, monkeypatch):
    from config.settings import settings
    target = tmp_path / "uploads"
    monkeypatch.setattr(settings, "upload_dir", target)
    ensure_cleanup_dir()
    assert target.exists()
    ensure_cleanup_dir()
    assert target.exists()


@pytest.mark.asyncio
async def test_background_cleanup_loop_sweeps():
    d = Path(mkdtemp())
    f = d / "old_file"
    f.write_text("old")

    mgr = FileCleanupManager(d, ttl_seconds=0)

    call_count = 0
    original_sweep = mgr._sweep_expired

    def sweep_once():
        nonlocal call_count
        call_count += 1
        original_sweep()
        if call_count >= 1:
            raise asyncio.CancelledError()

    mgr._sweep_expired = sweep_once

    with pytest.raises(asyncio.CancelledError):
        await mgr.background_cleanup_loop(interval_seconds=0)

    assert call_count == 1

    for entry in d.iterdir():
        entry.unlink()
    d.rmdir()
