from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from config.settings import settings
from runtime.orchestration.batch_manager import BatchManager, ALLOWED_BATCH_ENDPOINTS


def make_input_file(tmp_path: Path, lines: list[dict]) -> str:
    file_id = "file_test_batch_input"
    path = tmp_path / file_id
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return file_id


def wait_batch(manager: BatchManager, batch_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        batch = manager.get_batch(batch_id)
        if batch and batch["status"] in {"completed", "failed", "cancelled"}:
            return batch
        time.sleep(0.05)
    raise AssertionError(f"batch {batch_id} did not finish")


def make_manager(tmp_path: Path, monkeypatch) -> BatchManager:
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    return BatchManager(base_dir=tmp_path / "batches")


def test_allowed_endpoints():
    assert "/v1/chat/completions" in ALLOWED_BATCH_ENDPOINTS
    assert "/v1/responses" in ALLOWED_BATCH_ENDPOINTS
    assert "/v1/embeddings" in ALLOWED_BATCH_ENDPOINTS


def test_create_batch_completed(tmp_path, monkeypatch):
    file_id = make_input_file(tmp_path, [{"custom_id": "a", "body": {"model": "x"}}])
    manager = make_manager(tmp_path, monkeypatch)
    seen = {}

    def handler(body):
        seen["model"] = body.get("model")
        return {"choices": [{"message": {"content": "ok"}}]}

    batch = manager.create_batch(file_id, "/v1/chat/completions", "24h", handler)
    assert batch["status"] in {"validating", "in_progress"}
    assert batch["request_counts"]["total"] == 1

    finished = wait_batch(manager, batch["id"])
    assert finished["status"] == "completed"
    assert finished["request_counts"]["completed"] == 1
    assert finished["request_counts"]["failed"] == 0
    assert finished["output_file_id"].startswith("file_batch_out_")
    assert seen == {"model": "x"}

    output = (tmp_path / finished["output_file_id"]).read_text(encoding="utf-8").strip()
    line = json.loads(output)
    assert line["custom_id"] == "a"
    assert line["response"]["status_code"] == 200
    assert line["response"]["body"]["choices"][0]["message"]["content"] == "ok"


def test_create_batch_failed_requests(tmp_path, monkeypatch):
    file_id = make_input_file(
        tmp_path,
        [{"custom_id": "ok", "body": {}}, {"custom_id": "bad", "body": {}}],
    )
    manager = make_manager(tmp_path, monkeypatch)

    def handler(body):
        raise RuntimeError("boom")

    batch = manager.create_batch(file_id, "/v1/chat/completions", "24h", handler)
    finished = wait_batch(manager, batch["id"])
    assert finished["status"] == "completed"
    assert finished["request_counts"]["completed"] == 0
    assert finished["request_counts"]["failed"] == 2
    assert len(finished["errors"]) == 2

    output_lines = (tmp_path / finished["output_file_id"]).read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 2
    assert json.loads(output_lines[0])["error"]["message"] == "boom"


def test_create_batch_unsupported_endpoint(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    import pytest
    with pytest.raises(ValueError):
        manager.create_batch("none", "/v1/foo", "24h", lambda b: {})


def test_create_batch_missing_file(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    import pytest
    with pytest.raises(ValueError):
        manager.create_batch("file_nope", "/v1/chat/completions", "24h", lambda b: {})


def test_create_batch_invalid_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "file_bad"
    path.write_text("not json\n", encoding="utf-8")
    manager = make_manager(tmp_path, monkeypatch)
    batch = manager.create_batch("file_bad", "/v1/chat/completions", "24h", lambda b: {})
    assert batch["status"] == "failed"
    assert "line 1" in batch["error"]


def test_create_batch_missing_custom_id(tmp_path, monkeypatch):
    path = tmp_path / "file_nocustom"
    path.write_text('{"body": {}}\n', encoding="utf-8")
    manager = make_manager(tmp_path, monkeypatch)
    batch = manager.create_batch("file_nocustom", "/v1/chat/completions", "24h", lambda b: {})
    assert batch["status"] == "failed"
    assert "custom_id" in batch["error"]


def test_cancel_batch_in_progress(tmp_path, monkeypatch):
    file_id = make_input_file(tmp_path, [{"custom_id": "a", "body": {}}] * 50)
    manager = make_manager(tmp_path, monkeypatch)
    started = threading.Event()

    def handler(body):
        started.set()
        time.sleep(0.1)
        return {}

    batch = manager.create_batch(file_id, "/v1/chat/completions", "24h", handler)
    started.wait(timeout=5)
    result = manager.cancel_batch(batch["id"])
    assert result["status"] == "cancelling"
    final = wait_batch(manager, batch["id"])
    assert final["status"] in {"cancelled", "completed"}


def test_cancel_batch_missing(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    assert manager.cancel_batch("batch_missing") is None


def test_list_and_get_batch(tmp_path, monkeypatch):
    file_id = make_input_file(tmp_path, [{"custom_id": "a", "body": {}}])
    manager = make_manager(tmp_path, monkeypatch)
    batch = manager.create_batch(file_id, "/v1/chat/completions", "24h", lambda b: {})
    wait_batch(manager, batch["id"])
    assert manager.get_batch(batch["id"])["id"] == batch["id"]
    assert manager.get_batch("batch_missing") is None
    listed = manager.list_batches()
    assert any(b["id"] == batch["id"] for b in listed)


def test_persists_across_reload(tmp_path, monkeypatch):
    file_id = make_input_file(tmp_path, [{"custom_id": "a", "body": {}}])
    manager = make_manager(tmp_path, monkeypatch)
    batch = manager.create_batch(file_id, "/v1/chat/completions", "24h", lambda b: {})
    wait_batch(manager, batch["id"])
    reloaded = BatchManager(base_dir=tmp_path / "batches")
    assert reloaded.get_batch(batch["id"])["status"] == "completed"
