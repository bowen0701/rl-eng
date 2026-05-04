"""Rollout execution for Tic-Tac-Toe self-play."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from rl_eng.agents.tic_tac_toe_td import Agent
from rl_eng.evaluation import evaluate_against_baselines
from rl_eng.envs.tic_tac_toe import Environment, CROSS, CIRCLE, show_board
from rl_eng.learners.td_learner import TDLearner


@dataclass
class SelfPlayMetrics:
    """Aggregated self-play outcomes."""

    agent1_wins: int = 0
    agent2_wins: int = 0
    ties: int = 0


TRAIN_METRICS_FIELDNAMES = [
    "episode",
    "window_size",
    "agent1_wins",
    "agent2_wins",
    "ties",
    "agent1_win_rate",
    "agent2_win_rate",
    "tie_rate",
    "epsilon",
    "step_size",
]

EVAL_FIELDNAMES = [
    "episode",
    "trained_player",
    "opponent",
    "games",
    "wins",
    "losses",
    "ties",
    "win_rate",
    "loss_rate",
    "tie_rate",
]


def self_train(
    epochs: int = int(1e5),
    step_size: float = 0.01,
    epsilon: float = 0.01,
    print_per_epochs: int = 500,
    log_every: int = 500,
    eval_every: int = 5000,
    eval_episodes: int = 200,
    seed: Optional[int] = None,
    run_dir: Optional[str] = None,
    win_reward: float = 1.0,
    loss_reward: float = 0.0,
    tie_reward: float = 0.5,
) -> SelfPlayMetrics:
    """Run self-play rollouts and update agent value tables."""
    if seed is not None:
        np.random.seed(seed)

    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")

    agent1 = Agent(
        player="X",
        step_size=step_size,
        epsilon=epsilon,
        win_reward=win_reward,
        loss_reward=loss_reward,
        tie_reward=tie_reward,
    )
    agent2 = Agent(
        player="O",
        step_size=step_size,
        epsilon=epsilon,
        win_reward=win_reward,
        loss_reward=loss_reward,
        tie_reward=tie_reward,
    )
    agent1.init_state_value_table()
    agent2.init_state_value_table()

    learner = TDLearner(step_size=step_size)

    metrics = SelfPlayMetrics()
    window_metrics = SelfPlayMetrics()
    episodes_since_last_log = 0

    if run_dir:
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        _initialize_csv(os.path.join(run_dir, "train_metrics.csv"), TRAIN_METRICS_FIELDNAMES)
        _initialize_csv(os.path.join(run_dir, "eval_metrics.csv"), EVAL_FIELDNAMES)

    for epoch in range(1, epochs + 1):
        env = Environment()
        agent1.reset_episode()
        agent2.reset_episode()

        while not env.is_done():
            r1, c1, symbol1 = agent1.select_position(env)
            env = env.step(r1, c1, symbol1)
            learner.update(agent1.trajectory, agent1.model)

            if env.is_done():
                break

            r2, c2, symbol2 = agent2.select_position(env)
            env = env.step(r2, c2, symbol2)
            learner.update(agent2.trajectory, agent2.model)

        _finalize_terminal_transition(
            env=env,
            agent1=agent1,
            agent2=agent2,
            learner=learner,
            metrics_targets=(metrics, window_metrics),
        )
        episodes_since_last_log += 1

        if epoch % print_per_epochs == 0:
            print(
                "Epoch {}: Agent1 wins {}, Agent2 wins {}, ties {}".format(
                    epoch,
                    round(metrics.agent1_wins / epoch, 2),
                    round(metrics.agent2_wins / epoch, 2),
                    round(metrics.ties / epoch, 2),
                )
            )
            show_board(env)
            print("---")

        if run_dir and (epoch % log_every == 0 or epoch == epochs):
            _append_csv_row(
                os.path.join(run_dir, "train_metrics.csv"),
                TRAIN_METRICS_FIELDNAMES,
                _build_training_metrics_row(
                    episode=epoch,
                    window_size=episodes_since_last_log,
                    metrics=window_metrics,
                    epsilon=epsilon,
                    step_size=step_size,
                ),
            )
            window_metrics = SelfPlayMetrics()
            episodes_since_last_log = 0

        if run_dir and (epoch % eval_every == 0 or epoch == epochs):
            rng_state = np.random.get_state()
            eval_rows = evaluate_against_baselines(agent1, agent2, eval_episodes=eval_episodes, episode=epoch)
            np.random.set_state(rng_state)
            for row in eval_rows:
                _append_csv_row(os.path.join(run_dir, "eval_metrics.csv"), EVAL_FIELDNAMES, row)

    if run_dir:
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        agent1.save_state_value_table(checkpoint_dir)
        agent2.save_state_value_table(checkpoint_dir)

    return metrics


def _finalize_terminal_transition(
    env: Environment,
    agent1: Agent,
    agent2: Agent,
    learner: TDLearner,
    metrics_targets: Iterable[SelfPlayMetrics],
) -> None:
    """Apply the final backup needed after a terminal transition."""
    is_greedy = True
    for metrics in metrics_targets:
        _record_outcome(env, metrics)

    if env.winner == CROSS:
        agent2.add_state(env.state, is_greedy)
        learner.update(agent2.trajectory, agent2.model)
        return

    if env.winner == CIRCLE:
        agent1.add_state(env.state, is_greedy)
        learner.update(agent1.trajectory, agent1.model)
        return

    agent2.add_state(env.state, is_greedy)
    learner.update(agent2.trajectory, agent2.model)


def _record_outcome(env: Environment, metrics: SelfPlayMetrics) -> None:
    """Update metrics with the terminal game outcome."""
    if env.winner == CROSS:
        metrics.agent1_wins += 1
        return
    if env.winner == CIRCLE:
        metrics.agent2_wins += 1
        return
    metrics.ties += 1


def _build_training_metrics_row(
    episode: int,
    window_size: int,
    metrics: SelfPlayMetrics,
    epsilon: float,
    step_size: float,
) -> Dict[str, Any]:
    return {
        "episode": episode,
        "window_size": window_size,
        "agent1_wins": metrics.agent1_wins,
        "agent2_wins": metrics.agent2_wins,
        "ties": metrics.ties,
        "agent1_win_rate": round(metrics.agent1_wins / window_size, 6),
        "agent2_win_rate": round(metrics.agent2_wins / window_size, 6),
        "tie_rate": round(metrics.ties / window_size, 6),
        "epsilon": epsilon,
        "step_size": step_size,
    }


def _initialize_csv(path: str, fieldnames: Iterable[str]) -> None:
    """Create a CSV file and write the header row."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(path: str, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    """Append a CSV row using the provided field order."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
