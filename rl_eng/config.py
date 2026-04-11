from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Base configuration for RL training hyperparameters."""
    epochs: int = 100000
    step_size: float = 0.1
    epsilon: float = 0.01
    gamma: float = 1.0  # Discount factor
    win_reward: float = 1.0
    loss_reward: float = -1.0
    tie_reward: float = -0.1


@dataclass
class BaseConfig:
    """Base configuration for RL experiments."""
    env: str
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
