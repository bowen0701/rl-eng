"""Rollout execution for Tic-Tac-Toe self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rl_eng.agents.tic_tac_toe_td import Agent
from rl_eng.envs.tic_tac_toe import Environment, CROSS, CIRCLE, show_board
from rl_eng.learners.td import TDLearner


@dataclass
class SelfPlayMetrics:
    """Aggregated self-play outcomes."""

    agent1_wins: int = 0
    agent2_wins: int = 0
    ties: int = 0


def self_train(
    epochs: int = int(1e5),
    step_size: float = 0.01,
    epsilon: float = 0.01,
    print_per_epochs: int = 500,
    seed: Optional[int] = None,
    run_dir: Optional[str] = None,
    win_reward: float = 1.0,
    loss_reward: float = 0.0,
    tie_reward: float = 0.5,
) -> SelfPlayMetrics:
    """Run self-play rollouts and update agent value tables."""
    if seed is not None:
        np.random.seed(seed)

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
            env=env, agent1=agent1, agent2=agent2, learner=learner, metrics=metrics
        )

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

    if run_dir:
        agent1.save_state_value_table(run_dir)
        agent2.save_state_value_table(run_dir)

    return metrics


def _finalize_terminal_transition(
    env: Environment,
    agent1: Agent,
    agent2: Agent,
    learner: TDLearner,
    metrics: SelfPlayMetrics,
) -> None:
    """Apply the final backup needed after a terminal transition."""
    is_greedy = True
    if env.winner == CROSS:
        metrics.agent1_wins += 1
        agent2.add_state(env.state, is_greedy)
        learner.update(agent2.trajectory, agent2.model)
        return

    if env.winner == CIRCLE:
        metrics.agent2_wins += 1
        agent1.add_state(env.state, is_greedy)
        learner.update(agent1.trajectory, agent1.model)
        return

    metrics.ties += 1
    agent2.add_state(env.state, is_greedy)
    learner.update(agent2.trajectory, agent2.model)
