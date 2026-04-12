from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np

NMARKS: int = 3
BOARD_NROWS: int = 3
BOARD_NCOLS: int = 3
BOARD_SIZE: int = BOARD_NROWS * BOARD_NCOLS

CROSS: int = 1
CIRCLE: int = -1
EMPTY: int = 0

class Environment:
    """Environment class for Tic-Tac Toe."""

    def __init__(self) -> None:
        self.steps_left: int = BOARD_SIZE
        self.board: np.ndarray = (np.array([EMPTY] * BOARD_SIZE)
                                    .reshape((BOARD_NROWS, BOARD_NCOLS)))
        self.state: str = self._hash(self.board)
        self.winner: int = EMPTY

    @staticmethod
    def _hash(board: np.ndarray) -> str:
        """Computes a string hash of the board state."""
        return ','.join([str(x) for x in list(board.reshape(BOARD_SIZE))])

    def get_positions(self) -> List[Tuple[int, int]]:
        """Get possible action positions given current board."""
        positions: List[Tuple[int, int]] = []
        for r in range(BOARD_NROWS):
            for c in range(BOARD_NCOLS):
                if self.board[r][c] == EMPTY:
                    positions.append((r, c))
        return positions

    def is_done(self) -> bool:
        """Check if the game is done."""
        return self.steps_left == 0

    def _copy(self) -> Environment:
        """Copy to a new Environment instance."""
        env_copy = Environment()
        env_copy.steps_left = self.steps_left
        env_copy.board = self.board.copy()
        env_copy.state = self.state
        env_copy.winner = self.winner
        return env_copy

    def _judge(self) -> Optional[Environment]:
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

            diag1_arr, diag2_arr = np.array(diag1), np.array(diag2)
            if (np.sum(diag1_arr) == symbol * NMARKS or 
                np.sum(diag2_arr) == symbol * NMARKS):
                self.winner = symbol
                self.steps_left = 0
                return self
        return None

    def step(self, r: int, c: int, symbol: int) -> Environment:
        """Take a step with symbol."""
        env_next = self._copy()
        env_next.board[r][c] = symbol
        env_next.state = self._hash(env_next.board)
        env_next.steps_left -= 1
        env_next._judge()
        return env_next

    def show_board(self) -> None:
        """Show board."""
        board_list = self.board.tolist()
        printable_board: List[List[str]] = []
        for r in range(BOARD_NROWS):
            row: List[str] = []
            for c in range(BOARD_NCOLS):
                if board_list[r][c] == CROSS:
                    row.append('X')
                elif board_list[r][c] == CIRCLE:
                    row.append('O')
                else:
                    row.append(' ')
            printable_board.append(row)

        print('Board: is_done={}, steps_left={}, winner={}'
              .format(self.is_done(), self.steps_left, self.winner))
        for r in range(BOARD_NROWS):
            print(printable_board[r])

    @staticmethod
    def _dfs_states(cur_symbol: int, env: Environment, all_state_env_d: Dict[str, Environment]) -> None:
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
    def get_all_states(cls) -> Dict[str, Environment]:
        """Get all states from the init state."""
        # The player who plays first always uses 'X'.
        cur_symbol = CROSS

        # Apply DFS to collect all states.
        env = Environment()
        all_state_env_d: Dict[str, Environment] = dict()
        all_state_env_d[env.state] = env
        cls._dfs_states(cur_symbol, env, all_state_env_d)
        return all_state_env_d
