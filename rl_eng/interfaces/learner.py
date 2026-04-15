from abc import ABC, abstractmethod
from typing import Any


class Learner(ABC):
    """Abstract base class for optimizers (PPO, DPO, diffusion)."""

    @abstractmethod
    def update(self, data: Any, model: Any) -> None:
        """Perform a gradient or value-based update using collected trajectories."""
        pass
