from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml

from rl_eng.config import BaseConfig
from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE, EMPTY, NMARKS, BOARD_NROWS, BOARD_NCOLS, BOARD_SIZE, Environment
from rl_eng.envs.tic_tac_toe.utils import show_board
from rl_eng.agents.tic_tac_toe_td import Agent
from rl_eng.rollout.tic_tac_toe import self_train

@dataclass
class TicTacToeConfig(BaseConfig):
    """Specific configuration for Tic-Tac-Toe."""
    env: str = "tic_tac_toe"


def main():
    """Main entry point for Tic-Tac-Toe RL training.
    
    Usage:
        python3 -m experiments.tic_tac_toe.train --epochs 100000 --step_size 0.75 --epsilon 0.75 --seed 42
    """
    # Configuration lifecycle: 
    # 1. Initialize default config object to use for argparse defaults
    config = TicTacToeConfig()

    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Reinforcement Learning Training")
    
    # Train command arguments
    parser.add_argument("--epochs", type=int, default=config.training.epochs, help="Number of training epochs")
    parser.add_argument("--step_size", type=float, default=config.training.step_size, help="Learning step size")
    parser.add_argument("--epsilon", type=float, default=config.training.epsilon, help="Exploration rate")
    parser.add_argument("--log_every", type=int, default=config.training.log_every, help="Episodes per training metrics row")
    parser.add_argument("--eval_every", type=int, default=config.training.eval_every, help="Episodes per baseline evaluation")
    parser.add_argument("--eval_episodes", type=int, default=config.training.eval_episodes, help="Games per evaluation matchup")
    parser.add_argument("--seed", type=int, default=config.seed, help="Random seed for reproducibility")
    parser.add_argument("--win_reward", type=float, default=config.training.win_reward, help="Reward for winning")
    parser.add_argument("--loss_reward", type=float, default=config.training.loss_reward, help="Reward for losing")
    parser.add_argument("--tie_reward", type=float, default=config.training.tie_reward, help="Reward for a tie")

    args = parser.parse_args()

    # 2. Apply overrides from argparse
    config.seed = args.seed
    config.training.epochs = args.epochs
    config.training.step_size = args.step_size
    config.training.epsilon = args.epsilon
    config.training.log_every = args.log_every
    config.training.eval_every = args.eval_every
    config.training.eval_episodes = args.eval_episodes
    config.training.win_reward = args.win_reward
    config.training.loss_reward = args.loss_reward
    config.training.tie_reward = args.tie_reward

    # 3. Generate run_id and directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    run_id = f"{config.env}_{timestamp}_s{config.seed}_g{git_hash}"
    run_dir = os.path.join("experiments", "tic_tac_toe", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True) # Create checkpoints directory

    # 4. Freeze config and write to YAML
    with open(os.path.join(run_dir, "config.yml"), 'w') as f:
        yaml.dump(asdict(config), f, sort_keys=False)

    print(f"Starting training run: {run_id}")
    self_train(
        epochs=config.training.epochs,
        step_size=config.training.step_size,
        epsilon=config.training.epsilon,
        print_per_epochs=500,
        log_every=config.training.log_every,
        eval_every=config.training.eval_every,
        eval_episodes=config.training.eval_episodes,
        seed=config.seed,
        run_dir=run_dir,
        win_reward=config.training.win_reward,
        loss_reward=config.training.loss_reward,
        tie_reward=config.training.tie_reward
    )

if __name__ == '__main__':
    main()
