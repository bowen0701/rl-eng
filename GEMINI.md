# GEMINI.md - Project Context & Instructions

## Project Overview
**rl-eng** is a Reinforcement Learning (RL) engineering and research platform. It currently features a comprehensive implementation of Tic-Tac-Toe using Temporal Difference (TD) learning, specifically TD(0).

### Core Architecture
- **`rl_eng/`**: Contains the core RL logic and environment definitions.
  - `tic_tac_toe.py`: Implements the `Environment`, `Agent` (TD learner), and `Human` players.
  - `config.py`: Centralized configuration using Python dataclasses.
  - `multi_armed_bandits.py`: Implementations of MAB algorithms.
- **`games/tic_tac_toe/`**: A self-contained Pygame-based GUI application.
  - `gui.py`: The Pygame game loop and rendering logic.
  - `launcher.py`: The entry point for the GUI app (supports both development and bundled execution).
  - `build_app.sh`: Script to package the game into a standalone macOS `.app` bundle using PyInstaller.
- **`runs/`**: Stores training artifacts, including state-value tables (JSON) and run-specific configurations (YAML).
- **`notebooks/`**: Jupyter notebooks for interactive research and algorithm prototyping.

## Building and Running

### Core RL CLI
Run the core logic directly via the module interface:
- **Train an agent**:
  ```bash
  python3 -m rl_eng.tic_tac_toe train --epochs 100000 --step_size 0.75 --epsilon 0.75 --seed 42
  ```
- **Play (CLI mode)**:
  ```bash
  python3 -m rl_eng.tic_tac_toe play --run_id <run_id>
  ```

### Pygame GUI
- **Launch in development**:
  ```bash
  python3 games/tic_tac_toe/launcher.py --run_id <run_id>
  ```
- **Build standalone macOS App**:
  ```bash
  ./games/tic_tac_toe/build_app.sh --run_id <run_id>
  ```
  Artifacts will be in `games/tic_tac_toe/dist/TicTacToe.zip`.

### Testing & Linting
- **Run tests**: `pytest` (Note: ensure a `tests/` directory is created if adding new tests).
- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type Checking**: `mypy .`

## Development Conventions

### Coding Style
- **Python Version**: >= 3.9
- **Docstrings**: Follow **Google Style** docstrings.
- **Linting**: Enforced via **Ruff** with a line length of 127.
- **Types**: Use type hints where possible; `mypy` is configured for basic type safety.

### RL Mechanics
- **State Representation**: Tic-Tac-Toe states are represented as comma-separated strings of board values (1 for X, -1 for O, 0 for empty).
- **Value Function**: The agent uses a state-value table initialized via DFS of all reachable states.
- **Updates**: TD(0) update rule: `V(St) = V(St) + alpha * (V(St+1) - V(St))`.
- **Rewards**: Win: 1.0, Loss: 0.0 (or -1.0 depending on config), Tie: 0.5.

### CI/CD
- GitHub Actions are configured in `.github/workflows/`.
- Pre-commit hooks are defined in `.pre-commit-config.yaml` at the root.

## Key Files
- `pyproject.toml`: Project metadata and tool configurations (Ruff, Mypy, Pytest).
- `rl_eng/tic_tac_toe.py`: The "Source of Truth" for the game logic and RL agent.
- `games/tic_tac_toe/gui.py`: Handles all visual interactions and the Pygame event loop.
