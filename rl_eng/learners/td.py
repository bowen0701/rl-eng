"""Temporal-difference learners for value-based RL."""

from rl_eng.data.trajectory import Trajectory
from rl_eng.interfaces.learner import Learner
from rl_eng.models.state_value_table import StateValueTable


class TDLearner(Learner):
    """Temporal-difference learner for state-value tables."""

    def __init__(self, step_size: float) -> None:
        """Initialize the TD learner with a step size (alpha)."""
        self.step_size = step_size

    def update(self, data: Trajectory, model: StateValueTable) -> None:
        """Perform a TD(0) backup using the provided trajectory and value table.

        Args:
            data: The trajectory containing visited states and parent links.
            model: The StateValueTable model to update.
        """
        s = data.last_state
        if s is None:
            return

        # Traverse back the whole player's states to back up using parent links.
        while s in data.parent_by_state:
            s_par = data.parent_by_state[s]
            is_greedy = data.is_greedy_by_state[s]

            if is_greedy:
                v_s = model.forward(s)
                v_s_par = model.forward(s_par)
                # TD(0) update: V(s_par) <- V(s_par) + alpha * [V(s) - V(s_par)]
                new_v_s_par = v_s_par + self.step_size * (v_s - v_s_par)
                model.update_value(s_par, new_v_s_par)

            s = s_par
