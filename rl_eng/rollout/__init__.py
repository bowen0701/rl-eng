"""Rollout execution package."""

from rl_eng.rollout.trajectory import Trajectory, TrajectoryStep

__all__ = ["SelfPlayMetrics", "Trajectory", "TrajectoryStep", "self_train"]


def __getattr__(name: str):
    """Lazily resolve rollout exports to avoid package import cycles."""
    if name in {"SelfPlayMetrics", "self_train"}:
        from rl_eng.rollout.tic_tac_toe import SelfPlayMetrics, self_train

        exports = {
            "SelfPlayMetrics": SelfPlayMetrics,
            "self_train": self_train,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
