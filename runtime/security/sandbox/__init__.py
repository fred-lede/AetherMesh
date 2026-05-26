import sys

from runtime.security.sandbox.manager import SandboxManager
from runtime.security.sandbox.platform import SandboxResult, PlatformSandbox
from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile

if sys.platform == "darwin":
    from runtime.security.sandbox.mac_sandbox import MacSandbox
elif sys.platform == "linux":
    from runtime.security.sandbox.linux_sandbox import LinuxSandbox

__all__ = [
    "SandboxProfile",
    "SandboxResult",
    "PlatformSandbox",
    "SandboxManager",
    "builtin_profiles",
    "default_profile",
]

if sys.platform == "darwin":
    __all__.append("MacSandbox")
elif sys.platform == "linux":
    __all__.append("LinuxSandbox")
