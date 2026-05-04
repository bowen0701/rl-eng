"""
Usage:
    python3 -m pytest tests/test_rollout_tic_tac_toe.py
    python3 -m pytest tests
"""

import csv

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


def test_self_train_writes_metrics_and_eval_csv(tmp_path):
    metrics = self_train(
        epochs=5,
        print_per_epochs=1000,
        log_every=2,
        eval_every=3,
        eval_episodes=4,
        seed=1,
        run_dir=str(tmp_path),
    )

    assert metrics.agent1_wins + metrics.agent2_wins + metrics.ties == 5

    with open(tmp_path / "metrics.csv", "r", newline="") as f:
        metrics_rows = list(csv.DictReader(f))

    assert [int(row["episode"]) for row in metrics_rows] == [2, 4, 5]
    assert [int(row["window_size"]) for row in metrics_rows] == [2, 2, 1]

    with open(tmp_path / "eval.csv", "r", newline="") as f:
        eval_rows = list(csv.DictReader(f))

    assert len(eval_rows) == 8
    assert {int(row["episode"]) for row in eval_rows} == {3, 5}
    assert {row["opponent"] for row in eval_rows} == {"random", "rule_based"}
    assert {row["trained_player"] for row in eval_rows} == {"X", "O"}
    assert {int(row["games"]) for row in eval_rows} == {4}

    for row in eval_rows:
        total_rate = float(row["win_rate"]) + float(row["loss_rate"]) + float(row["tie_rate"])
        assert total_rate == 1.0
