import json
import os
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rl_eng.data import Trajectory
from rl_eng.envs.tic_tac_toe import CROSS, CIRCLE, Environment
from rl_eng.envs.tic_tac_toe.utils import get_all_states
from rl_eng.models.state_value_table import StateValueTable

if TYPE_CHECKING:
    from rl_eng.rollout.tic_tac_toe import SelfPlayMetrics

# TODO: Refactor Agent to Actor / Rollout (interacts with env).
# rl_eng/rollout/
#   actor.py
#   action_selection.py
#   trajectory_collector.py

class Agent:
    """Agent class for Tic-Tac-Toe game."""

    def __init__(self, 
                 player: str = 'X', 
                 step_size: Optional[float] = None, 
                 epsilon: Optional[float] = None, 
                 win_reward: float = 1.0, 
                 loss_reward: float = 0.0, 
                 tie_reward: float = 0.5) -> None:
        self.player: str = player
        if self.player == 'X':
            self.symbol: int = CROSS
        elif self.player == 'O':
            self.symbol: int = CIRCLE
        else:
            raise ValueError("Input player should be 'X' or 'O'")

        self.step_size: Optional[float] = step_size
        self.epsilon: Optional[float] = epsilon
        self.win_reward: float = win_reward
        self.loss_reward: float = loss_reward
        self.tie_reward: float = tie_reward

        # Use StateValueTable model.
        self.model = StateValueTable()

        # Memoize action state, its parent state & is_greedy bool.
        self.reset_episode()

    @property
    def V(self) -> Dict[str, float]:
        """Expose raw table for backward compatibility."""
        return self.model.table

    def init_state_value_table(self) -> None:
        """Init state-value table."""
        from rl_eng.envs.tic_tac_toe.utils import get_all_states
        all_state_env_d: Dict[str, Environment] = get_all_states()

        for s, env in all_state_env_d.items():
            if env.winner == self.symbol:
                # If agent is winner, it gets win_reward.
                self.model.update_value(s, self.win_reward)
            elif env.winner == -self.symbol:
                # If agent is loser, it gets loss_reward.
                self.model.update_value(s, self.loss_reward)
            else:
                # For tie or other cases, agent get tie_reward.
                self.model.update_value(s, self.tie_reward)

    def reset_episode(self) -> None:
        """Init episode."""
        self.trajectory = Trajectory()

    @property
    def states(self) -> List[str]:
        """Expose ordered states for compatibility with existing callers."""
        return self.trajectory.states

    @property
    def state_parent_d(self) -> Dict[str, str]:
        """Expose parent links for compatibility with existing callers."""
        return self.trajectory.parent_by_state

    @property
    def state_isgreedy_d(self) -> Dict[str, bool]:
        """Expose greedy flags for compatibility with existing callers."""
        return self.trajectory.is_greedy_by_state

    def _exploit_and_explore(self, env: Environment, positions: List[Tuple[int, int]]) -> Tuple[int, int, str, bool]:
        """Exploit and explore by the epsilon-greedy strategy."""
        p = np.random.random()
        if self.epsilon is not None and p > self.epsilon:
            # Exploit by selecting the move with the greatest value.
            val_positions: List[Tuple[float, Tuple[int, int]]] = []
            for (r, c) in positions:
                env_next = env.step(r, c, self.symbol)
                s = env_next.state
                v = self.model.forward(s)
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

    def add_state(self, state_next: str, is_greedy: bool) -> 'Agent':
        self.trajectory.add_step(state=state_next, is_greedy=is_greedy)
        return self

    def select_position(self, env: Environment) -> Tuple[int, int, int]:
        """Select a action position by the epsilon-greedy strategy."""
        # Get next action positions from environment.
        positions = env.get_actions()

        # Exloit and explore by the epsilon-greedy strategy.
        (r, c, state_next, is_greedy) = self._exploit_and_explore(
            env, positions)

        # Add state.
        self.add_state(state_next, is_greedy)
        return r, c, self.symbol

    def backup_state_value(self) -> None:
        """Back up value by a temporal-difference learning after a greedy move."""
        if self.step_size is None:
            return

        from rl_eng.learners.td_learner import TDLearner

        learner = TDLearner(step_size=self.step_size)
        learner.update(self.trajectory, self.model)

    def save_state_value_table(self, run_dir: str) -> None:
        """Save learned state-value table into the specified run directory."""
        filename = "state_values_x.json" if self.symbol == CROSS else "state_values_o.json"
        path = os.path.join(run_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.model.table, f)

    def load_state_value_table(self, run_dir: str) -> None:
        """Load learned state-value table from the specified run directory."""
        filename = "state_values_x.json" if self.symbol == CROSS else "state_values_o.json"
        path = os.path.join(run_dir, filename)
        with open(path, 'r') as f:
            self.model.table = json.load(f)


def self_train(
    epochs: int = int(1e5),
    step_size: float = 0.01,
    epsilon: float = 0.01,
    print_per_epochs: int = 500,
    seed: Optional[int] = None,
    run_dir: Optional[str] = None,
    win_reward: float = 1.0,
    loss_reward: float = 0.0,
    tie_reward: float = 0.5,
) -> "SelfPlayMetrics":
    """Compatibility wrapper for the rollout-owned self-play loop."""
    from rl_eng.rollout.tic_tac_toe import self_train as rollout_self_train

    return rollout_self_train(
        epochs=epochs,
        step_size=step_size,
        epsilon=epsilon,
        print_per_epochs=print_per_epochs,
        seed=seed,
        run_dir=run_dir,
        win_reward=win_reward,
        loss_reward=loss_reward,
        tie_reward=tie_reward,
    )
