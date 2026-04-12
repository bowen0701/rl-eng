from rl_eng.agents.tic_tac_toe_tabular import Agent
from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE

def test_agent_init():
    agent_x = Agent(player='X')
    assert agent_x.symbol == CROSS
    
    agent_o = Agent(player='O')
    assert agent_o.symbol == CIRCLE

def test_agent_value_init():
    agent = Agent(player='X', win_reward=1.0, loss_reward=0.0, tie_reward=0.5)
    agent.init_state_value_table()
    
    # Check if some states are in V
    assert len(agent.V) > 0
    # Initial state should be a tie_reward (0.5) since it's not a terminal win/loss
    assert agent.V["0,0,0,0,0,0,0,0,0"] == 0.5

def test_agent_add_state():
    agent = Agent(player='X')
    agent.add_state("state1", is_greedy=True)
    agent.add_state("state2", is_greedy=False)
    
    assert agent.states == ["state1", "state2"]
    assert agent.state_parent_d["state2"] == "state1"
    assert agent.state_isgreedy_d["state2"] is False
