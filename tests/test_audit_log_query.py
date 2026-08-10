from __future__ import annotations

import json
import time

from runtime.security.audit_log import AuditLog


def make_log(tmp_path, events):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    for action, actor, details, ts in events:
        with log._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": ts, "actor": actor, "action": action, "details": details}) + "\n")
    return log


def test_query_returns_all_in_order(tmp_path):
    log = make_log(
        tmp_path,
        [("a", "alice", {}, 1.0), ("b", "bob", {}, 2.0), ("c", "alice", {}, 3.0)],
    )
    events = log.query()
    assert [e["action"] for e in events] == ["a", "b", "c"]


def test_query_filter_action(tmp_path):
    log = make_log(
        tmp_path,
        [("a", "alice", {}, 1.0), ("b", "bob", {}, 2.0), ("a", "carol", {}, 3.0)],
    )
    events = log.query(action="a")
    assert [e["actor"] for e in events] == ["alice", "carol"]


def test_query_filter_actor(tmp_path):
    log = make_log(tmp_path, [("a", "alice", {}, 1.0), ("b", "bob", {}, 2.0)])
    events = log.query(actor="alice")
    assert len(events) == 1
    assert events[0]["action"] == "a"


def test_query_time_range(tmp_path):
    log = make_log(
        tmp_path,
        [("a", "alice", {}, 1.0), ("b", "bob", {}, 2.0), ("c", "carol", {}, 3.0)],
    )
    events = log.query(start_time=2.0, end_time=3.0)
    assert [e["action"] for e in events] == ["b", "c"]


def test_query_details_filter(tmp_path):
    log = make_log(
        tmp_path,
        [("tool_execution", "alice", {"tool_name": "shell", "is_error": False}, 1.0)],
    )
    events = log.query(details_filter={"tool_name": "shell"})
    assert len(events) == 1
    assert log.query(details_filter={"tool_name": "other"}) == []


def test_query_limit_offset(tmp_path):
    log = make_log(tmp_path, [(f"a{i}", "alice", {}, float(i)) for i in range(10)])
    events = log.query(limit=3, offset=2)
    assert [e["action"] for e in events] == ["a2", "a3", "a4"]


def test_query_missing_file(tmp_path):
    log = AuditLog(path=str(tmp_path / "nope.jsonl"))
    assert log.query() == []


def test_query_skips_bad_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"timestamp": 1, "action": "ok", "actor": "x", "details": {}}\nnot json\n', encoding="utf-8")
    log = AuditLog(path=str(path))
    events = log.query()
    assert len(events) == 1
    assert events[0]["action"] == "ok"


def test_record_writes_append(tmp_path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.record("login", {"method": "api_key"}, actor="alice")
    log.record("logout", {}, actor="alice")
    events = log.query()
    assert [e["action"] for e in events] == ["login", "logout"]
    assert events[0]["details"]["method"] == "api_key"


def test_recent_events_keeps_latest(tmp_path):
    log = make_log(tmp_path, [(f"a{i}", "alice", {}, float(i)) for i in range(10)])
    events = log.recent_events(limit=3)
    assert [e["action"] for e in events] == ["a7", "a8", "a9"]


def test_query_is_invoked_without_error_for_empty_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("", encoding="utf-8")
    log = AuditLog(path=str(path))
    assert log.query() == []
    assert time.time() > 0
