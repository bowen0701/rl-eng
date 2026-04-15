"""Utilities for Tic-Tac-Toe environment."""

from typing import Dict, List
from rl_eng.envs.tic_tac_toe.env import BOARD_NCOLS, BOARD_NROWS, CROSS, EMPTY, Environment


def show_board(env: Environment) -> None:
    """Show board."""
    board_list = env.board.tolist()
    printable_board: List[List[str]] = []
    for r in range(BOARD_NROWS):
        row: List[str] = []
        for c in range(BOARD_NCOLS):
            if board_list[r][c] == CROSS:
                row.append('X')
            elif board_list[r][c] == -CROSS:
                row.append('O')
            else:
                row.append(' ')
        printable_board.append(row)

    print('Board: is_done={}, steps_left={}, winner={}'
          .format(env.is_done(), env.steps_left, env.winner))
    for r in range(BOARD_NROWS):
        print(printable_board[r])


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
                        _dfs_states(-cur_symbol, env_next, all_state_env_d)


def get_all_states() -> Dict[str, Environment]:
    """Get all states from the init state."""
    cur_symbol = CROSS

    env = Environment()
    all_state_env_d: Dict[str, Environment] = dict()
    all_state_env_d[env.state] = env
    _dfs_states(cur_symbol, env, all_state_env_d)
    return all_state_env_d
