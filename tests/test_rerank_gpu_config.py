from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BAT = REPO_ROOT / "scripts" / "start-rerank-server.bat"
SH = REPO_ROOT / "scripts" / "start-rerank-server.sh"
CLUSTER = REPO_ROOT / "config" / "cluster.yaml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOC = REPO_ROOT / "docs" / "providers" / "rerank-deployment.md"

BGE_BLOB = "sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"


def _fake_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    (models / "blobs").mkdir(parents=True)
    (models / "blobs" / BGE_BLOB).write_bytes(b"fake")
    return models


def _parse(output: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"RESOLVED_(GPU|CVD)=(.*)$", line.strip())
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def _clean_env(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.pop("AIIH_RERANK_CUDA_DEVICES", None)
    env.pop("AIIH_RERANK_DEVICE", None)
    env.update(extra)
    return env


def _run_bat(tmp_path: Path, args: list[str], extra_env: dict[str, str]) -> dict[str, str]:
    models = _fake_models_dir(tmp_path)
    source = BAT.read_text(encoding="utf-8")
    patched, count = re.subn(
        r'^"%LLAMA_SERVER%".*$',
        'echo RESOLVED_GPU=%GPU%\r\necho RESOLVED_CVD=%CUDA_VISIBLE_DEVICES%\r\nexit /b 0',
        source,
        flags=re.MULTILINE,
    )
    assert count == 1, "expected exactly one llama-server invocation to patch"
    script = tmp_path / "rerank.bat"
    script.write_text(patched, encoding="utf-8", newline="")

    fake_binary = tmp_path / "llama-server.exe"
    fake_binary.write_bytes(b"")
    env = _clean_env(
        os.environ,
        {
            "OLLAMA_MODELS": str(models),
            "AIIH_LLAMA_SERVER": str(fake_binary),
            **extra_env,
        },
    )
    result = subprocess.run(
        ["cmd", "/c", str(script), *args],
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, f"bat failed: {result.stdout}\n{result.stderr}"
    return _parse(result.stdout)


def _bash_executable() -> str | None:
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _run_sh(tmp_path: Path, args: list[str], extra_env: dict[str, str]) -> dict[str, str]:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("no POSIX bash available to execute the .sh script")
    models = _fake_models_dir(tmp_path)
    source = SH.read_text(encoding="utf-8")
    marker = 'exec "$LLAMA_SERVER"'
    assert marker in source, "expected the .sh to exec llama-server"
    patched = (
        source[: source.index(marker)]
        + 'echo "RESOLVED_GPU=$GPU"\n'
        + 'echo "RESOLVED_CVD=$CUDA_VISIBLE_DEVICES"\n'
    )
    script = tmp_path / "rerank.sh"
    script.write_text(patched, encoding="utf-8", newline="\n")

    fake_binary = tmp_path / "llama-server"
    fake_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")

    env = _clean_env(
        os.environ,
        {
            "OLLAMA_MODELS": str(models),
            "AIIH_LLAMA_SERVER": str(fake_binary),
            **extra_env,
        },
    )
    result = subprocess.run(
        [bash, str(script), *args],
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, f"sh failed: {result.stdout}\n{result.stderr}"
    return _parse(result.stdout)


class TestBatchGpuResolution:
    def test_defaults_to_gpu_one(self, tmp_path: Path) -> None:
        resolved = _run_bat(tmp_path, ["bge-reranker-v2-m3", "11436"], {})
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"

    def test_third_argument_selects_the_gpu(self, tmp_path: Path) -> None:
        resolved = _run_bat(tmp_path, ["bge-reranker-v2-m3", "11436", "0"], {})
        assert resolved["GPU"] == "0"
        assert resolved["CVD"] == "0"

    def test_env_var_overrides_argument(self, tmp_path: Path) -> None:
        resolved = _run_bat(
            tmp_path,
            ["bge-reranker-v2-m3", "11436", "0"],
            {"AIIH_RERANK_CUDA_DEVICES": "1"},
        )
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"

    def test_deprecated_alias_overrides_argument(self, tmp_path: Path) -> None:
        resolved = _run_bat(
            tmp_path,
            ["bge-reranker-v2-m3", "11436", "0"],
            {"AIIH_RERANK_DEVICE": "1"},
        )
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"

    def test_only_the_selected_gpu_is_visible(self, tmp_path: Path) -> None:
        resolved = _run_bat(tmp_path, ["bge-reranker-v2-m3", "11436", "1"], {})
        assert resolved["CVD"] == "1"


class TestShellGpuResolution:
    def test_defaults_to_gpu_one(self, tmp_path: Path) -> None:
        resolved = _run_sh(tmp_path, ["bge-reranker-v2-m3", "11436"], {})
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"

    def test_third_argument_selects_the_gpu(self, tmp_path: Path) -> None:
        resolved = _run_sh(tmp_path, ["bge-reranker-v2-m3", "11436", "0"], {})
        assert resolved["GPU"] == "0"
        assert resolved["CVD"] == "0"

    def test_env_var_overrides_argument(self, tmp_path: Path) -> None:
        resolved = _run_sh(
            tmp_path,
            ["bge-reranker-v2-m3", "11436", "0"],
            {"AIIH_RERANK_CUDA_DEVICES": "1"},
        )
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"

    def test_deprecated_alias_overrides_argument(self, tmp_path: Path) -> None:
        resolved = _run_sh(
            tmp_path,
            ["bge-reranker-v2-m3", "11436", "0"],
            {"AIIH_RERANK_DEVICE": "1"},
        )
        assert resolved["GPU"] == "1"
        assert resolved["CVD"] == "1"


class TestScriptMainGpuArgument:
    @pytest.mark.parametrize("script", [BAT, SH])
    def test_main_gpu_is_zero(self, script: Path) -> None:
        text = script.read_text(encoding="utf-8")
        assert re.search(r"-mg\s+0", text), (
            "with CUDA_VISIBLE_DEVICES set, the only visible GPU is index 0, "
            "so -mg must be 0"
        )

    @pytest.mark.parametrize("script", [BAT, SH])
    def test_no_hardcoded_visible_devices(self, script: Path) -> None:
        text = script.read_text(encoding="utf-8")
        assert not re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*\d", text), (
            "CUDA_VISIBLE_DEVICES must be derived from the resolved GPU, "
            "never hardcoded to a fixed ordinal"
        )


class TestClusterConfig:
    def test_rerank_port_declares_the_4070_gpu(self) -> None:
        import yaml

        data = yaml.safe_load(CLUSTER.read_text(encoding="utf-8"))
        workers = {int(w["port"]): w for w in data["local_workers"]}
        rerank = workers[11436]
        assert rerank["gpu_id"] == 1, (
            "port 11436 serves the standalone llama-server rerank and must "
            "declare gpu_id 1 (RTX 4070 Ti SUPER), not the RTX 5090"
        )
        assert rerank["role"] == "rerank"


class TestEnvExample:
    def test_documents_primary_cuda_devices_variable(self) -> None:
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "AIIH_RERANK_CUDA_DEVICES=1" in text

    def test_marks_legacy_device_variable_deprecated(self) -> None:
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        idx = text.find("AIIH_RERANK_DEVICE")
        assert idx != -1
        window = text[max(0, idx - 500) : idx + 500]
        assert "deprecated" in window.lower(), (
            "AIIH_RERANK_DEVICE must be marked deprecated in .env.example"
        )


class TestDeploymentDoc:
    def test_documents_resolution_order(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        assert "AIIH_RERANK_CUDA_DEVICES" in text
        assert "cluster.yaml" in text

    def test_explains_visible_devices_isolation(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES" in text
