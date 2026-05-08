from __future__ import annotations

from dataclasses import dataclass, field

from runtime.state.execution_state import AgentStatus, validate_transition


@dataclass
class AgentMachineState:
    status: AgentStatus = AgentStatus.IDLE
    agent_id: str = ""
    current_task: str = ""
    step_count: int = 0
    max_steps: int = 25

    def transition(self, target: AgentStatus) -> bool:
        if validate_transition(self.status, target):
            self.status = target
            return True
        return False
