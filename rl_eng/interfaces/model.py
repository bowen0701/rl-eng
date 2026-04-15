from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Abstract base class for all neural networks or value tables (pure functions)."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass given input states."""
        pass
