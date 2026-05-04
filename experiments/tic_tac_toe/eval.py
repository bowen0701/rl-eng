from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
from dataclasses import dataclass
from typing import Tuple

import yaml

from rl_eng.config import BaseConfig
from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE, EMPTY, NMARKS, BOARD_NROWS, BOARD_NCOLS, BOARD_SIZE, Environment
from rl_eng.envs.tic_tac_toe.utils import show_board
from rl_eng.agents.tic_tac_toe_td import Agent

@dataclass
class TicTacToeConfig(BaseConfig):
    """Specific configuration for Tic-Tac-Toe."""
    env: str = "tic_tac_toe"

class Human:
    """Human class for Tic-Tac-Toe game."""

    def __init__(self, player: str = 'X') -> None:
        self.player: str = player
        if self.player == 'X':
            self.symbol: int = CROSS
        elif self.player == 'O':
            self.symbol: int = CIRCLE

    def select_position(self, env: Environment) -> Tuple[int, int, int]:
        """Select a position given current state."""
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


def human_agent_compete(run_dir: str, config: TicTacToeConfig) -> None:
    """Human compete with agent."""
    human_name = input('Please input your name:\n')
    while True:
        human_player = input(
            'Please input your player: 1st player (X), 2nd player (O):\n')
        if human_player in ['X', 'O']:
            break

    env = Environment()
    show_board(env)
    print('---')

    if human_player == 'X':
        human, agent = Human(player='X'), Agent(player='O', epsilon=0.0)
        player1, player2 = human, agent
        player1_name, player2_name = human_name, 'Robot'
    else:
        agent, human = Agent(player='X', epsilon=0.0), Human(player='O')
        player1, player2 = agent, human
        player1_name, player2_name = 'Robot', human_name

    agent.load_state_value_table(os.path.join(run_dir, "checkpoints"))

    while not env.is_done():
        r1, c1, symbol1 = player1.select_position(env)
        env = env.step(r1, c1, symbol1)
        print('Player1, {} ({}), puts ({}, {})'
              .format(player1_name, player1.player, r1, c1))
        show_board(env)
        print('---')

        if env.is_done():
            break

        r2, c2, symbol2 = player2.select_position(env)
        env = env.step(r2, c2, symbol2)
        print('Player2, {} ({}), puts ({}, {})'
              .format(player2_name, player2.player, r2, c2))
        show_board(env)
        print('---')

    if env.winner == human.symbol:
        print('Congrats {}, you win!'.format(human_name))
    elif env.winner == -human.symbol:
        print('{} loses to Robot...'.format(human_name))
    else:
        print('{} and Robot tie.'.format(human_name))


def main():
    """Main entry point for Tic-Tac-Toe RL evaluation and play.

    Usage:
        python3 -m experiments.tic_tac_toe.eval play --run_id <run_id>
    """
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Reinforcement Learning Evaluation/Play")
    subparsers = parser.add_subparsers(dest="cmd", help="Command to run")
    play_parser = subparsers.add_parser("play", help="Play against the trained RL agent")
    play_parser.add_argument("--run_id", type=str, required=True, help="Run ID to load the agent from")

    args = parser.parse_args()

    run_dir = os.path.join("experiments", "tic_tac_toe", "runs", args.run_id)
    if not os.path.exists(run_dir):
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    config = TicTacToeConfig()
    config_path = os.path.join(run_dir, "config.yml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
            config.seed = raw_config.get("seed", config.seed)

    config.training.epsilon = 0.0

    human_agent_compete(run_dir=run_dir, config=config)

if __name__ == '__main__':
    main()
