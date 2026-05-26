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

logger = logging.getLogger("sandbox.linux")

_CLONE_NEWNET = 0x40000000
_CLONE_NEWNS = 0x00020000


class LinuxSandbox(PlatformSandbox):
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        scratch = self._create_scratch_dir()
        workdir = cwd or scratch

        start = time.monotonic()
        try:
            preexec = lambda: self._apply_linux_limits(profile)  # noqa: E731
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
            return SandboxResult(output="", return_code=-1, duration_ms=duration, timed_out=True)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(output=str(e), return_code=-1, duration_ms=duration)
        finally:
            self._cleanup_scratch(scratch)

    def _apply_linux_limits(self, profile: SandboxProfile) -> None:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(profile.max_cpu_time_sec), int(profile.max_cpu_time_sec)),
        )
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (profile.max_memory_bytes, hard))
        if not profile.allow_network:
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.unshare(_CLONE_NEWNET)
            except Exception:
                logger.warning("unshare(CLONE_NEWNET) failed — network isolation not enforced")

    def _create_scratch_dir(self) -> str:
        return tempfile.mkdtemp(prefix="aether_sandbox_")

    def _cleanup_scratch(self, path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean up scratch dir %s", path)
