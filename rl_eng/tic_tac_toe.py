from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import yaml

from rl_eng.config import BaseConfig


NMARKS = 3
BOARD_NROWS = BOARD_NCOLS = 3
BOARD_SIZE = BOARD_NROWS * BOARD_NCOLS

CROSS = 1
CIRCLE = -1
EMPTY = 0


@dataclass
class TicTacToeConfig(BaseConfig):
    """Specific configuration for Tic-Tac-Toe."""
    env: str = "tic_tac_toe"


class Environment:
    """Environment class for Tic-Tac Toe."""

    def __init__(self):
        self.steps_left = BOARD_SIZE
        self.board = (np.array([EMPTY] * BOARD_SIZE)
                        .reshape((BOARD_NROWS, BOARD_NCOLS)))
        self.state = self._hash(self.board)
        self.winner = EMPTY

    @staticmethod
    def _hash(board):
        return ','.join([str(x) for x in list(board.reshape(BOARD_SIZE))])

    def get_positions(self):
        """Get possible action positions given current board."""
        positions = []
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if self.board[r][c] == EMPTY:
                    positions.append((r, c))
        return positions

    def is_done(self):
        """Check the game is done."""
        return self.steps_left == 0

    def _copy(self):
        """Copy to a new Environment instance."""
        env_copy = Environment()
        env_copy.steps_left = self.steps_left
        env_copy.board = self.board.copy()
        env_copy.state = self.state
        env_copy.winner = self.winner
        return env_copy

    def _judge(self):
        """Judge winner based on the current board."""
        # Check rows.
        for r in range(BOARD_NROWS):
            row = self.board[r, :]
            symbol = row[0]
            if symbol != EMPTY and np.sum(row) == symbol * NMARKS:
                self.winner = symbol
                self.steps_left = 0
                return self

        # Check columns.
        for c in range(BOARD_NCOLS):
            col = self.board[:, c]
            symbol = col[0]
            if symbol != EMPTY and np.sum(col) == symbol * NMARKS:
                self.winner = symbol
                self.steps_left = 0
                return self

        # Check diagonals.
        mid = BOARD_NROWS // 2
        symbol = self.board[mid][mid]
        if symbol != EMPTY: 
            diag1, diag2 = [], []
            for i in range(BOARD_NROWS):
                diag1.append(self.board[i][i])
                diag2.append(self.board[i][BOARD_NROWS - i - 1])

            diag1, diag2 = np.array(diag1), np.array(diag2)
            if (np.sum(diag1) == symbol * NMARKS or 
                np.sum(diag2) == symbol * NMARKS):
                self.winner = symbol
                self.steps_left = 0
                return self

    def step(self, r, c, symbol):
        """Take a step with symbol."""
        env_next = self._copy()
        env_next.board[r][c] = symbol
        env_next.state = self._hash(env_next.board)
        env_next.steps_left -= 1
        env_next._judge()
        return env_next

    def show_board(self):
        """Show board."""
        board = self.board.tolist()
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if board[r][c] == CROSS:
                    board[r][c] = 'X'
                elif board[r][c] == CIRCLE:
                    board[r][c] = 'O'
                else:
                    board[r][c] = ' '

        print('Board: is_done={}, steps_left={}, winner={}'
              .format(self.is_done(), self.steps_left, self.winner))
        for r in range(BOARD_NROWS):
            print(board[r])

    @staticmethod
    def _dfs_states(cur_symbol, env, all_state_env_d):
        """DFS for next state by recursion."""
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if env.board[r][c] == EMPTY:
                    env_next = env.step(r, c, cur_symbol)
                    if env_next.state not in all_state_env_d:
                        all_state_env_d[env_next.state] = env_next

                        # If game is not ended, continue DFS.
                        if not env_next.is_done():
                            Environment._dfs_states(-cur_symbol, env_next, all_state_env_d)

    @classmethod
    def get_all_states(cls):
        """Get all states from the init state."""
        # The player who plays first always uses 'X'.
        cur_symbol = CROSS

        # Apply DFS to collect all states.
        env = Environment()
        all_state_env_d = dict()
        all_state_env_d[env.state] = env
        cls._dfs_states(cur_symbol, env, all_state_env_d)
        return all_state_env_d


class Agent:
    """Agent class for Tic-Tac-Toe game."""

    def __init__(self, player='X', step_size=None, epsilon=None, win_reward=1.0, loss_reward=0.0, tie_reward=0.5):
        self.player = player
        if self.player == 'X':
            self.symbol = CROSS
        elif self.player == 'O':
            self.symbol = CIRCLE
        else:
            raise ValueError("Input player should be 'X' or 'O'")

        self.step_size = step_size
        self.epsilon = epsilon
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.tie_reward = tie_reward

        # Create a state-value table V:state->value.
        self.V = dict()

        # Memoize action state, its parent state & is_greedy bool:
        # state_parent_d:state->parent state & state_isgreedy_d:state->is_greedy bool.
        self.reset_episode()

    def init_state_value_table(self):
        """Init state-value table."""
        all_state_env_d = Environment.get_all_states()

        for s, env in all_state_env_d.items():
            if env.winner == self.symbol:
                # If agent is winner, it gets win_reward.
                self.V[s] = self.win_reward
            elif env.winner == -self.symbol:
                # If agent is loser, it gets loss_reward.
                self.V[s] = self.loss_reward
            else:
                # For tie or other cases, agent get tie_reward.
                self.V[s] = self.tie_reward

    def reset_episode(self):
        """Init episode."""
        self.states = []
        self.state_parent_d = dict()
        self.state_isgreedy_d = dict()

    def _exploit_and_explore(self, env, positions):
        """Exploit and explore by the epsilon-greedy strategy:

        Procedure:
          - Take exploratory moves in the p% of times. 
          - Take greedy moves in the (100-p)% of times.
        where p% is epsilon. 
        If epsilon is zero, then always use greedy strategy.
        """
        p = np.random.random()
        if p > self.epsilon:
            # Exploit by selecting the move with the greatest value.
            val_positions = []
            for (r, c) in positions:
                env_next = env.step(r, c, self.symbol)
                s = env_next.state
                v = self.V[s]
                val_positions.append((v, (r, c)))

            # Break ties randomly: shuffle & sort.
            np.random.shuffle(val_positions)
            val_positions.sort(key=lambda x: x[0], reverse=True)
            (r, c) = val_positions[0][1]
            is_greedy = True
        else:
            # Explore by selecting randomly from among moves.
            np.random.shuffle(positions)
            n = len(positions)
            (r, c) = positions[np.random.randint(n)]
            is_greedy = False

        env_next = env.step(r, c, self.symbol)
        state_next = env_next.state
        return (r, c, state_next, is_greedy)

    def add_state(self, state_next, is_greedy):
        if self.states:
            state = self.states[-1]
            self.state_parent_d[state_next] = state
        self.state_isgreedy_d[state_next] = is_greedy
        self.states.append(state_next)
        return self

    def select_position(self, env):
        """Select a action position by the epsilon-greedy strategy.

        Agent gets candidate positions given current state.
        Then it selects a position by strategy.
        """
        # Get next action positions from environment.
        positions = env.get_positions()

        # Exloit and explore by the epsilon-greedy strategy.
        (r, c, state_next, is_greedy) = self._exploit_and_explore(
            env, positions)

        # Add state.
        self.add_state(state_next, is_greedy)
        return r, c, self.symbol

    def backup_state_value(self):
        """Back up value by a temporal-difference learning after a greedy move.

        Temporal-difference learning:
          V(S_t) <- V(S_t) + a * [V(S_{t+1}) - V(S_t)]
        where a is the step size, and V(S_t) is the state-value function
        at time step t.
        """
        s = self.states[-1]

        # Traverse back the whole player's states to back up.
        while s in self.state_parent_d:
            s_par = self.state_parent_d[s]
            is_greedy = self.state_isgreedy_d[s]
            if is_greedy:
                self.V[s_par] += self.step_size * (self.V[s] - self.V[s_par])
            s = s_par

    def save_state_value_table(self, run_dir):
        """Save learned state-value table into the specified run directory."""
        filename = "state_values_x.json" if self.symbol == CROSS else "state_values_o.json"
        path = os.path.join(run_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.V, f)

    def load_state_value_table(self, run_dir):
        """Load learned state-value table from the specified run directory."""
        filename = "state_values_x.json" if self.symbol == CROSS else "state_values_o.json"
        path = os.path.join(run_dir, filename)
        with open(path, 'r') as f:
            self.V = json.load(f)


def self_train(epochs=int(1e5), step_size=0.01, epsilon=0.01, print_per_epochs=500, seed=None, run_dir=None, 
               win_reward=1.0, loss_reward=0.0, tie_reward=0.5):
    """Self train an agent by playing games against itself."""
    if seed is not None:
        np.random.seed(seed)

    agent1 = Agent(player='X', step_size=step_size, epsilon=epsilon, 
                   win_reward=win_reward, loss_reward=loss_reward, tie_reward=tie_reward)
    agent2 = Agent(player='O', step_size=step_size, epsilon=epsilon,
                   win_reward=win_reward, loss_reward=loss_reward, tie_reward=tie_reward)
    agent1.init_state_value_table()
    agent2.init_state_value_table()

    n_agent1_wins = 0
    n_agent2_wins = 0
    n_ties = 0

    for i in range(1, epochs + 1):
        # Reset both agents after epoch was done.
        env = Environment()
        agent1.reset_episode()
        agent2.reset_episode()

        while not env.is_done():
            # Agent 1 plays one step.
            r1, c1, symbol1 = agent1.select_position(env)
            env = env.step(r1, c1, symbol1)
            agent1.backup_state_value()

            if env.is_done():
                break

            # Agent 2 plays the next step.
            r2, c2, symbol2 = agent2.select_position(env)
            env = env.step(r2, c2, symbol2)
            agent2.backup_state_value()

        # Set final state with is_greedy=True to backup loser's value.
        is_greedy = True
        if env.winner == CROSS:
            n_agent1_wins += 1

            agent2.add_state(env.state, is_greedy)
            agent2.backup_state_value()
        elif env.winner == CIRCLE:
            n_agent2_wins += 1

            agent1.add_state(env.state, is_greedy)
            agent1.backup_state_value()
        else:
            n_ties += 1

            # In a tie, Agent 1 just moved (9th move), Agent 2 needs to back up.
            agent2.add_state(env.state, is_greedy)
            agent2.backup_state_value()

        # Print board.
        if i % print_per_epochs == 0:
            print('Epoch {}: Agent1 wins {}, Agent2 wins {}, ties {}'
                  .format(i,
                          round(n_agent1_wins / i, 2), 
                          round(n_agent2_wins / i, 2), 
                          round(n_ties / i, 2)))
            env.show_board()
            print('---')

    if run_dir:
        agent1.save_state_value_table(run_dir)
        agent2.save_state_value_table(run_dir)


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
