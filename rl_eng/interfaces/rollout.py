from abc import ABC, abstractmethod
from typing import Any


class Rollout(ABC):
    """Abstract base class for sampling + execution engine."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute rollouts in the given environments with provided models."""
        pass
