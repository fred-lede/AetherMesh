from __future__ import annotations

from dataclasses import dataclass, field

from runtime.state.execution_state import ProviderStatus, validate_transition


@dataclass
class ProviderMachineState:
    status: ProviderStatus = ProviderStatus.IDLE
    selected_provider: str = ""
    selected_model: str = ""
    retry_count: int = 0
    cooldown_until: float = 0.0

    def transition(self, target: ProviderStatus) -> bool:
        if validate_transition(self.status, target):
            self.status = target
            return True
        return False
