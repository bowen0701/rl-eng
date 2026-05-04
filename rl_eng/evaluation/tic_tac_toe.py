"""Evaluation helpers for Tic-Tac-Toe policies."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from rl_eng.agents.tic_tac_toe_td import Agent
from rl_eng.envs.tic_tac_toe import BOARD_NCOLS, BOARD_NROWS, CIRCLE, CROSS, EMPTY, Environment


class RandomPolicy:
    """Uniform-random Tic-Tac-Toe policy."""

    def __init__(self, player: str) -> None:
        self.player = player
        self.symbol = CROSS if player == "X" else CIRCLE

    def select_position(self, env: Environment) -> Tuple[int, int, int]:
        """Sample a legal move uniformly at random."""
        positions = env.get_actions()
        index = int(np.random.randint(len(positions)))
        r, c = positions[index]
        return r, c, self.symbol


class RuleBasedPolicy:
    """Simple fixed policy with win/block/center/corner priorities."""

    def __init__(self, player: str) -> None:
        self.player = player
        self.symbol = CROSS if player == "X" else CIRCLE
        self.opponent_symbol = -self.symbol

    def select_position(self, env: Environment) -> Tuple[int, int, int]:
        """Choose a legal move from a small deterministic priority list."""
        for symbol in (self.symbol, self.opponent_symbol):
            winning_move = self._find_winning_move(env, symbol)
            if winning_move is not None:
                r, c = winning_move
                return r, c, self.symbol

        for r, c in self._priority_moves(env):
            return r, c, self.symbol

        raise ValueError("RuleBasedPolicy received an environment with no legal actions")

    def _find_winning_move(self, env: Environment, symbol: int) -> Optional[Tuple[int, int]]:
        for r, c in env.get_actions():
            if env.step(r, c, symbol).winner == symbol:
                return r, c
        return None

    def _priority_moves(self, env: Environment) -> List[Tuple[int, int]]:
        center = [(BOARD_NROWS // 2, BOARD_NCOLS // 2)]
        corners = [(0, 0), (0, BOARD_NCOLS - 1), (BOARD_NROWS - 1, 0), (BOARD_NROWS - 1, BOARD_NCOLS - 1)]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        legal = set(env.get_actions())
        ordered_moves = center + corners + edges
        return [move for move in ordered_moves if move in legal]


def clone_greedy_agent(agent: Agent) -> Agent:
    """Create an evaluation-only copy with exploration disabled."""
    eval_agent = Agent(
        player=agent.player,
        epsilon=0.0,
        win_reward=agent.win_reward,
        loss_reward=agent.loss_reward,
        tie_reward=agent.tie_reward,
    )
    eval_agent.model.table = dict(agent.model.table)
    return eval_agent


def evaluate_against_baselines(agent_x: Agent, agent_o: Agent, eval_episodes: int, episode: int) -> List[Dict[str, Any]]:
    """Evaluate the current policies against fixed baselines."""
    baselines: List[Tuple[str, Callable[[str], Any]]] = [
        ("random", RandomPolicy),
        ("rule_based", RuleBasedPolicy),
    ]
    rows: List[Dict[str, Any]] = []

    for opponent_name, baseline_factory in baselines:
        rows.append(
            _build_eval_row(
                episode=episode,
                trained_player="X",
                opponent=opponent_name,
                counts=_play_matchup(
                    player_x=clone_greedy_agent(agent_x),
                    player_o=baseline_factory("O"),
                    episodes=eval_episodes,
                    trained_symbol=CROSS,
                ),
            )
        )
        rows.append(
            _build_eval_row(
                episode=episode,
                trained_player="O",
                opponent=opponent_name,
                counts=_play_matchup(
                    player_x=baseline_factory("X"),
                    player_o=clone_greedy_agent(agent_o),
                    episodes=eval_episodes,
                    trained_symbol=CIRCLE,
                ),
            )
        )

    return rows


def _play_matchup(player_x: Any, player_o: Any, episodes: int, trained_symbol: int) -> Dict[str, int]:
    wins = 0
    losses = 0
    ties = 0

    for _ in range(episodes):
        env = Environment()
        _reset_policy_episode(player_x)
        _reset_policy_episode(player_o)
        while not env.is_done():
            r_x, c_x, symbol_x = player_x.select_position(env)
            env = env.step(r_x, c_x, symbol_x)
            if env.is_done():
                break

            r_o, c_o, symbol_o = player_o.select_position(env)
            env = env.step(r_o, c_o, symbol_o)

        if env.winner == EMPTY:
            ties += 1
        elif env.winner == trained_symbol:
            wins += 1
        else:
            losses += 1

    return {"wins": wins, "losses": losses, "ties": ties}


def _reset_policy_episode(policy: Any) -> None:
    reset_episode = getattr(policy, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _build_eval_row(episode: int, trained_player: str, opponent: str, counts: Dict[str, int]) -> Dict[str, Any]:
    games = counts["wins"] + counts["losses"] + counts["ties"]
    return {
        "episode": episode,
        "trained_player": trained_player,
        "opponent": opponent,
        "games": games,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "ties": counts["ties"],
        "win_rate": round(counts["wins"] / games, 6),
        "loss_rate": round(counts["losses"] / games, 6),
        "tie_rate": round(counts["ties"] / games, 6),
    }
