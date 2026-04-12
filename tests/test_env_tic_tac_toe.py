"""
Usage:
    # From project root:
    python3 -m pytest tests/test_env_tic_tac_toe.py
"""
from rl_eng.envs.tic_tac_toe import Environment, CROSS, CIRCLE, EMPTY

def test_initial_state():
    env = Environment()
    assert env.steps_left == 9
    assert env.winner == EMPTY
    assert env.is_done() is False
    assert env.state == "0,0,0,0,0,0,0,0,0"

def test_step():
    env = Environment()
    env_next = env.step(0, 0, CROSS)
    assert env_next.board[0, 0] == CROSS
    assert env_next.steps_left == 8
    assert env_next.state != env.state

def test_win_detection_rows():
    env = Environment()
    env = env.step(0, 0, CROSS)
    env = env.step(0, 1, CROSS)
    env = env.step(0, 2, CROSS)
    assert env.winner == CROSS
    assert env.is_done() is True

def test_win_detection_cols():
    env = Environment()
    env = env.step(0, 0, CIRCLE)
    env = env.step(1, 0, CIRCLE)
    env = env.step(2, 0, CIRCLE)
    assert env.winner == CIRCLE
    assert env.is_done() is True

def test_win_detection_diagonals():
    # Diagonal 1
    env = Environment()
    env = env.step(0, 0, CROSS)
    env = env.step(1, 1, CROSS)
    env = env.step(2, 2, CROSS)
    assert env.winner == CROSS
    
    # Diagonal 2
    env = Environment()
    env = env.step(0, 2, CIRCLE)
    env = env.step(1, 1, CIRCLE)
    env = env.step(2, 0, CIRCLE)
    assert env.winner == CIRCLE

def test_tie_game():
    env = Environment()
    # Fill board without a win
    # X O X
    # X X O
    # O X O
    moves = [
        (0,0,1), (0,1,-1), (0,2,1),
        (1,0,1), (1,1,1),  (1,2,-1),
        (2,0,-1), (2,1,1), (2,2,-1)
    ]
    for r, c, s in moves:
        env = env.step(r, c, s)
    
    assert env.is_done() is True
    assert env.winner == EMPTY
