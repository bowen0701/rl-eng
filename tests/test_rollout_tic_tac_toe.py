"""
Usage:
    python3 -m pytest tests/test_rollout_tic_tac_toe.py
    python3 -m pytest tests
"""

from rl_eng.agents.tic_tac_toe_td import self_train as agent_self_train
from rl_eng.data import Trajectory
from rl_eng.rollout.tic_tac_toe import SelfPlayMetrics, self_train


def test_rollout_self_train_returns_metrics():
    metrics = self_train(epochs=2, print_per_epochs=1000, seed=1)

    assert isinstance(metrics, SelfPlayMetrics)
    assert metrics.agent1_wins + metrics.agent2_wins + metrics.ties == 2


def test_agent_self_train_shim_uses_rollout_package():
    metrics = agent_self_train(epochs=1, print_per_epochs=1000, seed=1)

    assert isinstance(metrics, SelfPlayMetrics)


def test_trajectory_records_parent_links():
    trajectory = Trajectory()

    trajectory.add_step("state1", is_greedy=True)
    trajectory.add_step("state2", is_greedy=False)

    assert trajectory.states == ["state1", "state2"]
    assert trajectory.parent_by_state["state2"] == "state1"
    assert trajectory.is_greedy_by_state["state2"] is False
