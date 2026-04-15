"""Trajectory primitives for rollout execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TrajectoryStep:
    """A single visited state within a rollout trajectory."""

    state: str
    is_greedy: bool


@dataclass
class Trajectory:
    """Ordered episode states plus parent links for TD-style backup."""

    steps: List[TrajectoryStep] = field(default_factory=list)
    parent_by_state: Dict[str, str] = field(default_factory=dict)

    def add_step(self, state: str, is_greedy: bool) -> None:
        """Append a visited state and wire its parent to the previous state."""
        if self.steps:
            self.parent_by_state[state] = self.steps[-1].state
        self.steps.append(TrajectoryStep(state=state, is_greedy=is_greedy))

    @property
    def states(self) -> List[str]:
        """Return ordered state hashes for compatibility with legacy callers."""
        return [step.state for step in self.steps]

    @property
    def is_greedy_by_state(self) -> Dict[str, bool]:
        """Return per-state greedy flags for compatibility with TD backups."""
        return {step.state: step.is_greedy for step in self.steps}

    @property
    def last_state(self) -> Optional[str]:
        """Return the most recent state, if any."""
        if not self.steps:
            return None
        return self.steps[-1].state
