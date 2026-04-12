from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import yaml

from rl_eng.config import BaseConfig
from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE, EMPTY, NMARKS, BOARD_NROWS, BOARD_NCOLS, BOARD_SIZE, Environment
from rl_eng.agents.tic_tac_toe_tabular import Agent, self_train

@dataclass
class TicTacToeConfig(BaseConfig):
    """Specific configuration for Tic-Tac-Toe."""
    env: str = "tic_tac_toe"


class Human:
    """Human class for Tic-Tac-Toe game."""

    def __init__(self, player='X'):
        self.player = player
        if self.player == 'X':
            self.symbol = CROSS
        elif self.player == 'O':
            self.symbol = CIRCLE

    def select_position(self, env):
        """Select a position given current state."""
        # Get human input position.
        input_position_re = re.compile('^[0-2],[0-2]$')
        positions = set(env.get_positions())
        while True:
            input_position = input(
                'Please input position for {} in the format: "row,col" with '
                .format(self.player) +
                'row/col: 0~{}:\n'.format(BOARD_NROWS - 1))

            if not input_position_re.match(input_position):
                print('Input position style is incorrect!\n')
                continue

            input_position = tuple([int(x) for x in input_position.split(',')])
            if input_position in positions:
                break
            else:
                print('Input position was occupied, please input "row,col" again!\n')

        (r, c) = input_position
        return r, c, self.symbol


def human_agent_compete(run_dir, config):
    """Human compete with agent."""
    # Get human player.
    human_name = input('Please input your name:\n')
    while True:
        human_player = input('Please input your player: ' + 
                             '1st player (X), 2nd player (O):\n')
        if human_player in ['X', 'O']:
            break

    env = Environment()
    env.show_board()
    print('---')

    # Set up human & agent as player1 or player2.
    if human_player == 'X':
        # Player1: human, player2: agent.
        human, agent = Human(player='X'), Agent(player='O', epsilon=0.0)
        player1, player2 = human, agent
        player1_name, player2_name = human_name, 'Robot'
    else:
        # Player1: agent, player2: human.
        agent, human = Agent(player='X', epsilon=0.0), Human(player='O')
        player1, player2 = agent, human
        player1_name, player2_name = 'Robot', human_name

    agent.load_state_value_table(run_dir)

    # Start competition.
    while not env.is_done():
        # Player1 plays one step.
        r1, c1, symbol1 = player1.select_position(env)
        env = env.step(r1, c1, symbol1)
        print('Player1, {} ({}), puts ({}, {})'
              .format(player1_name, player1.player, r1, c1))
        env.show_board()
        print('---')

        if env.is_done():
            break

        # Player2 plays the next step.
        r2, c2, symbol2 = player2.select_position(env)
        env = env.step(r2, c2, symbol2)
        print('Player2, {} ({}), puts ({}, {})'
              .format(player2_name, player2.player, r2, c2))
        env.show_board()
        print('---')

    # Judge the winner.
    if env.winner == human.symbol:
        print('Congrats {}, you win!'.format(human_name))
    elif env.winner == -human.symbol:
        print('{} loses to Robot...'.format(human_name))
    else:
        print('{} and Robot tie.'.format(human_name))


def main():
    """Main entry point for Tic-Tac-Toe RL.
    
    Usage:
        # To train the agent:
        python3 -m rl_eng.tic_tac_toe train --epochs 100000 --step_size 0.75 --epsilon 0.75 --seed 42
        
        # To play against the agent:
        python3 -m rl_eng.tic_tac_toe play --run_id <run_id>
    """
    # Configuration lifecycle: 
    # 1. Initialize default config object to use for argparse defaults
    config = TicTacToeConfig()

    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Reinforcement Learning")
    subparsers = parser.add_subparsers(dest="cmd", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument("--epochs", type=int, default=config.training.epochs, help="Number of training epochs")
    train_parser.add_argument("--step_size", type=float, default=config.training.step_size, help="Learning step size")
    train_parser.add_argument("--epsilon", type=float, default=config.training.epsilon, help="Exploration rate")
    train_parser.add_argument("--seed", type=int, default=config.seed, help="Random seed for reproducibility")
    train_parser.add_argument("--win_reward", type=float, default=config.training.win_reward, help="Reward for winning")
    train_parser.add_argument("--loss_reward", type=float, default=config.training.loss_reward, help="Reward for losing")
    train_parser.add_argument("--tie_reward", type=float, default=config.training.tie_reward, help="Reward for a tie")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against the trained RL agent")
    play_parser.add_argument("--run_id", type=str, required=True, help="Run ID to load the agent from")

    args = parser.parse_args()

    # 2. Apply overrides from argparse
    if args.cmd == "train":
        config.seed = args.seed
        config.training.epochs = args.epochs
        config.training.step_size = args.step_size
        config.training.epsilon = args.epsilon
        config.training.win_reward = args.win_reward
        config.training.loss_reward = args.loss_reward
        config.training.tie_reward = args.tie_reward

        # 3. Generate run_id and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_id = f"{config.env}_{timestamp}_s{config.seed}"
        run_dir = os.path.join("runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        # 4. Freeze config and write to YAML
        with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
            yaml.dump(asdict(config), f, sort_keys=False)

        print(f"Starting training run: {run_id}")
        self_train(
            epochs=config.training.epochs,
            step_size=config.training.step_size,
            epsilon=config.training.epsilon,
            print_per_epochs=500,
            seed=config.seed,
            run_dir=run_dir,
            win_reward=config.training.win_reward,
            loss_reward=config.training.loss_reward,
            tie_reward=config.training.tie_reward
        )
    elif args.cmd == "play":
        run_dir = os.path.join("runs", args.run_id)
        if not os.path.exists(run_dir):
            print(f"Error: Run directory {run_dir} does not exist.")
            return

        # Load config from the run directory
        config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
                config.seed = raw_config.get("seed", config.seed)
                if "training" in raw_config:
                    config.training.epochs = raw_config["training"].get("epochs", config.training.epochs)
                    config.training.step_size = raw_config["training"].get("step_size", config.training.step_size)
                    config.training.epsilon = raw_config["training"].get("epsilon", config.training.epsilon)
                    config.training.win_reward = raw_config["training"].get("win_reward", config.training.win_reward)
                    config.training.loss_reward = raw_config["training"].get("loss_reward", config.training.loss_reward)
                    config.training.tie_reward = raw_config["training"].get("tie_reward", config.training.tie_reward)

        # Force epsilon to 0.0 for the competition
        config.training.epsilon = 0.0
        human_agent_compete(run_dir=run_dir, config=config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
