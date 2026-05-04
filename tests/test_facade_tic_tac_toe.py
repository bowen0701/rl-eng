"""
Usage:
    python3 -m pytest tests/test_facade_tic_tac_toe.py
    python3 -m pytest tests
"""
import experiments.tic_tac_toe as tic_tac_toe

def test_facade_constants():
    """Verify that board constants are still accessible via the facade."""
    assert tic_tac_toe.CROSS == 1
    assert tic_tac_toe.CIRCLE == -1
    assert tic_tac_toe.EMPTY == 0
    assert tic_tac_toe.NMARKS == 3

def test_facade_classes():
    """Verify that main classes are still accessible via the facade."""
    env = tic_tac_toe.Environment()
    assert isinstance(env, tic_tac_toe.Environment)

    agent = tic_tac_toe.Agent(player='X')
    assert isinstance(agent, tic_tac_toe.Agent)

    human = tic_tac_toe.Human(player='O')
    assert isinstance(human, tic_tac_toe.Human)

def test_facade_functions():
    """Verify that core functions are still accessible via the facade."""
    assert callable(tic_tac_toe.self_train)
    assert callable(tic_tac_toe.human_agent_compete)
