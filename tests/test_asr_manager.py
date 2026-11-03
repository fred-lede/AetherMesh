# tests/test_asr_manager.py
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

import runtime.asr.asr_manager as asr_manager_module
from runtime.asr.asr_manager import ASRConfigError, ASRManager


def _fake_adapter(name: str = "fake") -> MagicMock:
    return MagicMock(provider_name=name)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(asr_manager_module.settings, "asr_enabled", True)
    monkeypatch.setattr(asr_manager_module.settings, "asr_provider", "faster_whisper")
    monkeypatch.setattr(asr_manager_module.settings, "asr_model", "large-v3")
    monkeypatch.setattr(asr_manager_module.settings, "asr_device", "cuda")
    monkeypatch.setattr(asr_manager_module.settings, "asr_device_index", 0)
    monkeypatch.setattr(type(asr_manager_module.settings), "load_asr_config", lambda self: {})
    monkeypatch.setattr(asr_manager_module, "discover_gpus", lambda: [
        {"id": 0, "name": "GPU0", "memory_free_mb": 1000},
        {"id": 1, "name": "GPU1", "memory_free_mb": 2000},
    ])
    monkeypatch.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [])
    return monkeypatch


def _make_manager(builders: dict):
    persisted: list[dict] = []
    manager = ASRManager(builders=builders, dispose=MagicMock(), persist=persisted.append)
    return manager, persisted


def _wait_state(manager: ASRManager, state: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.snapshot()["state"] == state:
            return
        time.sleep(0.01)
    pytest.fail(f"state never reached '{state}': {manager.snapshot()}")


def _both_builders(builder):
    return {"local_cuda": builder, "local_cpu": builder}


def test_initial_state_idle():
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda data: None)
    snapshot = manager.snapshot()
    assert snapshot == {"state": "idle", "active": None, "requested": None, "error": None, "version": 0}
    assert manager.get_adapter() is None


def test_preload_disabled(env):
    env.setattr(asr_manager_module.settings, "asr_enabled", False)
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    manager.preload()
    assert manager.snapshot()["state"] == "disabled"
    assert manager.get_adapter() is None


def test_preload_success(env):
    adapter = _fake_adapter("a")
    manager, persisted = _make_manager(_both_builders(lambda t, p, m: adapter))
    manager.preload()
    _wait_state(manager, "ready")
    snapshot = manager.snapshot()
    assert snapshot["version"] == 1
    assert manager.get_adapter() is adapter
    assert snapshot["active"]["target_id"] == "local:cuda:0"
    assert snapshot["active"]["model"] == "large-v3"
    assert snapshot["active"]["provider"] == "faster_whisper"
    assert persisted == []


def test_preload_uses_persisted_config(env):
    env.setattr(type(asr_manager_module.settings), "load_asr_config", lambda self: {
        "target_id": "local:cpu", "provider": "faster_whisper", "model": "small",
    })
    seen = {}

    def builder(target, provider, model):
        seen["args"] = (target.target_id, target.kind, provider, model)
        return _fake_adapter("b")

    manager, _ = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "ready")
    assert seen["args"] == ("local:cpu", "local_cpu", "faster_whisper", "small")


def test_preload_failure_marks_failed(env):
    def builder(target, provider, model):
        raise RuntimeError("CUDA out of memory")

    manager, _ = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "failed")
    snapshot = manager.snapshot()
    assert "CUDA out of memory" in snapshot["error"]
    assert manager.get_adapter() is None
    assert snapshot["version"] == 0


def test_reload_success_swaps_and_disposes_old(env):
    adapter_a, adapter_b = _fake_adapter("a"), _fake_adapter("b")
    builders = {"local_cuda": lambda t, p, m: adapter_a, "local_cpu": lambda t, p, m: adapter_b}
    manager, persisted = _make_manager(builders)
    manager.preload()
    _wait_state(manager, "ready")

    snapshot = manager.reload("local:cpu", provider="qwen_asr", model="Qwen/Qwen3-ASR-1.7B")
    assert snapshot["state"] == "loading"
    _wait_state(manager, "ready")

    snapshot = manager.snapshot()
    assert snapshot["version"] == 2
    assert manager.get_adapter() is adapter_b
    assert snapshot["active"]["provider"] == "qwen_asr"
    assert snapshot["active"]["target_id"] == "local:cpu"
    manager._dispose.assert_called_once_with(adapter_a)
    assert persisted[-1] == {
        "target_id": "local:cpu",
        "provider": "qwen_asr",
        "model": "Qwen/Qwen3-ASR-1.7B",
        "updated_at": persisted[-1]["updated_at"],
    }
    assert persisted[-1]["updated_at"]


def test_reload_failure_rolls_back(env):
    adapter_a = _fake_adapter("a")
    calls = {"n": 0}

    def builder(target, provider, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return adapter_a
        raise RuntimeError("CUDA out of memory")

    manager, persisted = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "ready")
    manager.reload("local:cpu", provider="faster_whisper", model="medium")
    _wait_state(manager, "failed")

    snapshot = manager.snapshot()
    assert snapshot["version"] == 1
    assert manager.get_adapter() is adapter_a
    assert "CUDA out of memory" in snapshot["error"]
    assert snapshot["active"]["model"] == "large-v3"
    assert persisted[-1]["model"] == "medium"


def test_reload_concurrent_returns_409(env):
    adapter = _fake_adapter("a")
    gate = threading.Event()

    def slow_builder(target, provider, model):
        gate.wait(timeout=5)
        return adapter

    manager, _ = _make_manager(_both_builders(slow_builder))
    manager.reload("local:cuda:0", provider="faster_whisper", model="large-v3")
    _wait_state(manager, "loading")
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="faster_whisper", model="small")
    assert excinfo.value.status_code == 409
    gate.set()
    _wait_state(manager, "ready")


def test_reload_race_validation_vs_check_yields_single_worker(env):
    node_calls = {"n": 0}

    def node_builder(target, provider, model):
        node_calls["n"] += 1
        return _fake_adapter("node")

    fetch_entered = threading.Event()
    fetch_gate = threading.Event()
    build_gate = threading.Event()

    def gated_fetch():
        fetch_entered.set()
        fetch_gate.wait(timeout=5)
        return [
            {"node_id": "x", "ip": "10.0.0.1", "metadata": {"asr_worker": {"port": 9450, "healthy": True}}},
        ]

    env.setattr(asr_manager_module, "_fetch_cluster_nodes", gated_fetch)

    def slow_cuda_builder(target, provider, model):
        build_gate.wait(timeout=5)
        return _fake_adapter("cuda")

    builders = {
        "local_cuda": slow_cuda_builder,
        "local_cpu": lambda t, p, m: _fake_adapter("cpu"),
        "node": node_builder,
    }
    manager, persisted = _make_manager(builders)

    result: dict = {}

    def caller_a():
        try:
            manager.reload("node:x", provider=None, model=None)
            result["error"] = None
        except ASRConfigError as exc:
            result["error"] = exc

    thread = threading.Thread(target=caller_a, daemon=True)
    thread.start()
    assert fetch_entered.wait(timeout=5)

    snapshot = manager.reload("local:cuda:0", provider="faster_whisper", model=None)
    assert snapshot["state"] == "loading"

    fetch_gate.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(result["error"], ASRConfigError)
    assert result["error"].status_code == 409
    assert node_calls["n"] == 0
    assert len(persisted) == 1

    build_gate.set()
    _wait_state(manager, "ready")
    snapshot = manager.snapshot()
    assert snapshot["version"] == 1
    assert node_calls["n"] == 0


def test_reload_unknown_target_400(env):
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("warp-drive", provider=None, model=None)
    assert excinfo.value.status_code == 400


def test_reload_unknown_provider_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="mozart", model=None)
    assert excinfo.value.status_code == 400
    assert "valid_providers" in excinfo.value.details


def test_reload_unknown_model_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="qwen_asr", model="tiny-cn")
    assert excinfo.value.status_code == 400
    assert "Qwen/Qwen3-ASR-1.7B" in excinfo.value.details["valid_models"]


def test_reload_device_index_out_of_range_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cuda:9", provider="faster_whisper", model=None)
    assert excinfo.value.status_code == 400
    assert excinfo.value.details["valid_gpu_ids"] == [0, 1]


def test_reload_node_resolves_base_url_and_reports_worker_engine(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}}},
    ])
    built = {}
    remote = MagicMock(provider_name="remote")
    remote.health_check.return_value = {"reachable": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}

    def node_builder(target, provider, model):
        built["base_url"] = target.base_url
        built["node_id"] = target.node_id
        return remote

    manager, _ = _make_manager({"local_cuda": lambda t, p, m: None, "local_cpu": lambda t, p, m: None, "node": node_builder})
    manager.reload("node:m4", provider=None, model=None)
    _wait_state(manager, "ready")
    assert built["base_url"] == "http://10.1.1.5:9450"
    assert built["node_id"] == "m4"
    snapshot = manager.snapshot()
    assert snapshot["active"]["provider"] == "qwen_asr"
    assert snapshot["active"]["target_id"] == "node:m4"


def test_reload_node_without_worker_400(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": False}}},
    ])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("node:m4", provider=None, model=None)
    assert excinfo.value.status_code == 400


def test_reload_node_unknown_node_404(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [{"node_id": "other", "ip": "1.2.3.4", "metadata": {}}])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("node:ghost", provider=None, model=None)
    assert excinfo.value.status_code == 404


def test_reload_node_worker_health_unreachable_rolls_back(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "x", "ip": "10.9.9.9", "metadata": {"asr_worker": {"port": 9450, "healthy": True}}},
    ])
    remote = MagicMock(provider_name="remote_asr")
    remote.health_check.return_value = {"reachable": False}
    manager, _ = _make_manager({"node": lambda t, p, m: remote, "local_cuda": lambda t, p, m: None, "local_cpu": lambda t, p, m: None})
    manager.reload("node:x", provider=None, model=None)
    _wait_state(manager, "failed")
    assert manager.get_adapter() is None
    assert "unreachable" in manager.snapshot()["error"]


def test_list_targets_combines_sources(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}}},
        {"node_id": "bare", "ip": "10.1.1.6", "metadata": {}},
    ])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    targets = manager.list_targets()
    assert targets["local"][0] == {"id": "local:cuda:0", "name": "GPU0", "memory_free_mb": 1000}
    assert targets["local"][-1] == {"id": "local:cpu", "name": "CPU", "memory_free_mb": None}
    by_id = {node["id"]: node for node in targets["nodes"]}
    assert by_id["node:m4"]["reachable"] is True
    assert by_id["node:m4"]["engine"] == "qwen_asr"
    assert by_id["node:bare"]["reachable"] is False
    assert targets["nodes_unreachable"] is False


def test_list_targets_control_plane_down_degrades(env):
    def boom():
        raise requests.ConnectionError("control-plane down")

    env.setattr(asr_manager_module, "_fetch_cluster_nodes", boom)
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    targets = manager.list_targets()
    assert targets["nodes_unreachable"] is True
    assert targets["nodes"] == []
    assert [t["id"] for t in targets["local"]] == ["local:cuda:0", "local:cuda:1", "local:cpu"]


def test_get_adapter_during_loading_returns_previous(env):
    adapter_a = _fake_adapter("a")
    gate = threading.Event()

    def slow_builder(target, provider, model):
        gate.wait(timeout=5)
        return _fake_adapter("b")

    builders = {"local_cuda": lambda t, p, m: adapter_a, "local_cpu": slow_builder}
    manager, _ = _make_manager(builders)
    manager.preload()
    _wait_state(manager, "ready")
    manager.reload("local:cpu", provider="faster_whisper", model="small")
    _wait_state(manager, "loading")
    assert manager.get_adapter() is adapter_a
    gate.set()
    _wait_state(manager, "ready")
