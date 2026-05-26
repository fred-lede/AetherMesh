from __future__ import annotations

import logging
import os
import resource
import shutil
import subprocess
import tempfile
import time

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile

logger = logging.getLogger("sandbox.mac")


class MacSandbox(PlatformSandbox):
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        exec_env = os.environ.copy()
        exec_env["NO_PROXY"] = "*"
        exec_env["HTTP_PROXY"] = ""
        exec_env["HTTPS_PROXY"] = ""
        exec_env["http_proxy"] = ""
        exec_env["https_proxy"] = ""

        if env:
            exec_env.update(env)

        scratch = self._create_scratch_dir()
        workdir = cwd or scratch

        start = time.monotonic()
        try:
            preexec = lambda: self._apply_resource_limits(profile)  # noqa: E731
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=profile.timeout_sec,
                cwd=workdir,
                env=exec_env,
                preexec_fn=preexec,
                stdin=subprocess.PIPE if stdin else None,
                input=stdin,
            )
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(
                output=(result.stdout + result.stderr)[:100_000],
                return_code=result.returncode,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(
                output="",
                return_code=-1,
                duration_ms=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(
                output=str(e),
                return_code=-1,
                duration_ms=duration,
            )
        finally:
            self._cleanup_scratch(scratch)

    def _apply_resource_limits(self, profile: SandboxProfile) -> None:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(profile.max_cpu_time_sec), int(profile.max_cpu_time_sec)),
        )
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (profile.max_memory_bytes, hard))
        except (ValueError, PermissionError):
            logger.warning("RLIMIT_AS not supported in child process on this platform")

    def _create_scratch_dir(self) -> str:
        return tempfile.mkdtemp(prefix="aether_sandbox_")

    def _cleanup_scratch(self, path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean up scratch dir %s", path)
