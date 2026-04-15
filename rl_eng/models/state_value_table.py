"""State-value table model implementation."""

from typing import Any, Dict, Optional

from rl_eng.interfaces.model import Model


class StateValueTable(Model):
    """A simple dictionary-backed state-value table."""

    def __init__(self, initial_values: Optional[Dict[str, float]] = None) -> None:
        """Initialize the table with optional starting values."""
        self._table: Dict[str, float] = initial_values or {}

    def forward(self, state: str) -> float:
        """Look up the value for a given state. Returns 0.0 if unknown."""
        return self._table.get(state, 0.0)

    def update_value(self, state: str, value: float) -> None:
        """Update the value for a specific state."""
        self._table[state] = value

    @property
    def table(self) -> Dict[str, float]:
        """Expose the underlying dictionary for serialization."""
        return self._table

    @table.setter
    def table(self, values: Dict[str, float]) -> None:
        """Set the underlying dictionary."""
        self._table = values
