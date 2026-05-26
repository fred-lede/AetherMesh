from __future__ import annotations

import time
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

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
    mgr.cleanup_request("req_nonexistent")


def test_cleanup_partial_failure_logs_warning(tmp_dir):
    f1 = tmp_dir / "file_a"
    f1.write_text("a")

    mgr = FileCleanupManager(tmp_dir)
    mgr.track("req_1", ["file_a", "file_missing"])
    with patch("runtime.tools.file_cleanup.logger.warning") as mock_warn:
        mgr.cleanup_request("req_1")
        mock_warn.assert_called_once()


def test_ttl_sweep_removes_expired(tmp_dir):
    f1 = tmp_dir / "old_file"
    f1.write_text("old")

    mgr = FileCleanupManager(tmp_dir, ttl_seconds=0)
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
    mgr.track_current("file_a")
