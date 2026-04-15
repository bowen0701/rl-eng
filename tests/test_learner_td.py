"""Tests for TDLearner."""

import pytest
from rl_eng.data.trajectory import Trajectory
from rl_eng.learners.td import TDLearner


from rl_eng.models.state_value_table import StateValueTable

def test_td_learner_update():
    """Verify that TDLearner correctly updates a state-value table."""
    # Setup
    learner = TDLearner(step_size=0.1)
    trajectory = Trajectory()
    value_table = StateValueTable({"state_a": 0.5, "state_b": 1.0})

    # Add steps: state_a -> state_b (greedy)
    trajectory.add_step(state="state_a", is_greedy=True)
    trajectory.add_step(state="state_b", is_greedy=True)

    # Perform update
    learner.update(trajectory, value_table)

    # Expected value: V(state_a) <- 0.5 + 0.1 * (1.0 - 0.5) = 0.55
    assert value_table.forward("state_a") == pytest.approx(0.55)


def test_td_learner_no_greedy_no_update():
    """Verify that non-greedy moves do not trigger a value backup."""
    learner = TDLearner(step_size=0.1)
    trajectory = Trajectory()
    value_table = StateValueTable({"state_a": 0.5, "state_b": 1.0})

    # Add steps: state_a -> state_b (NOT greedy)
    trajectory.add_step(state="state_a", is_greedy=True)
    trajectory.add_step(state="state_b", is_greedy=False)

    learner.update(trajectory, value_table)

    # V(state_a) should remain unchanged
    assert value_table.forward("state_a") == 0.5


def test_td_learner_empty_trajectory():
    """Verify that updating with an empty trajectory does nothing."""
    learner = TDLearner(step_size=0.1)
    trajectory = Trajectory()
    value_table = StateValueTable({"state_a": 0.5})

    learner.update(trajectory, value_table)
    assert value_table.forward("state_a") == 0.5
