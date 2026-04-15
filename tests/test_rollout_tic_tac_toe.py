"""
Usage:
    python3 -m pytest tests/test_rollout_tic_tac_toe.py
    python3 -m pytest tests
"""

from rl_eng.agents.tic_tac_toe_td import self_train as agent_self_train
from rl_eng.rollout.tic_tac_toe import SelfPlayMetrics, self_train


def test_rollout_self_train_returns_metrics():
    metrics = self_train(epochs=2, print_per_epochs=1000, seed=1)

    assert isinstance(metrics, SelfPlayMetrics)
    assert metrics.agent1_wins + metrics.agent2_wins + metrics.ties == 2


def test_agent_self_train_shim_uses_rollout_package():
    metrics = agent_self_train(epochs=1, print_per_epochs=1000, seed=1)

    assert isinstance(metrics, SelfPlayMetrics)
