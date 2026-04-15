from abc import ABC, abstractmethod
from typing import Any, List


class Environment(ABC):
    """Abstract base class for all environment backends."""

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> "Environment":
        """Execute one time step in the environment."""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the current episode has ended."""
        pass

    @abstractmethod
    def get_actions(self) -> List[Any]:
        """Return the list of legal actions in the current state."""
        pass
