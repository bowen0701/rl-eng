import json
import os
import numpy as np

from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE, Environment

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
        """Exploit and explore by the epsilon-greedy strategy."""
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
        """Select a action position by the epsilon-greedy strategy."""
        # Get next action positions from environment.
        positions = env.get_positions()

        # Exloit and explore by the epsilon-greedy strategy.
        (r, c, state_next, is_greedy) = self._exploit_and_explore(
            env, positions)

        # Add state.
        self.add_state(state_next, is_greedy)
        return r, c, self.symbol

    def backup_state_value(self):
        """Back up value by a temporal-difference learning after a greedy move."""
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
