"""
# Interfaces diagram for subsystems.

                ┌──────────────┐
                │   scripts    │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │    infra     │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   rollout    │  ⭐ system heart
                └──────┬───────┘
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        envs         models        data
                       ↓
                ┌──────────────┐
                │  learners    │
                └──────────────┘
"""
