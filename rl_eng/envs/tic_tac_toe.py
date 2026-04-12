import numpy as np

NMARKS = 3
BOARD_NROWS = BOARD_NCOLS = 3
BOARD_SIZE = BOARD_NROWS * BOARD_NCOLS

CROSS = 1
CIRCLE = -1
EMPTY = 0

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
