# Repository Guidelines

## Project Structure & Module Organization
`rl_eng/` contains the core Python package, designed for scalability and future offline RL support:
- `rl_eng/envs/`: Environment logic and interaction backends.
- `rl_eng/agents/`: RL agent implementations (e.g., TD, PPO).
- `rl_eng/rollout/`: Sampling and execution engine.
- `rl_eng/data/`: Data primitives, trajectories, and datasets for online/offline RL.
- `rl_eng/config.py`: Common configurations and type-safe experiment settings.
- `rl_eng/tic_tac_toe.py`: Tic-Tac-Toe CLI entrypoint.
- `rl_eng/multi_armed_bandits.py`: Bandit experiment entrypoint.

Other key directories:
- `artifacts/apps/tic_tac_toe/`: Pygame launcher and macOS packaging scripts.
- `notebooks/`: Interactive exploration and analysis.
- `tests/`: Project tests mirroring the main package structure.
- `runs/`: Runtime artifacts (configs, state-values).
- `artifacts/`: Promoted exports, checkpoints, and evaluation reports (treat as generated).

## Build, Test, and Development Commands
Install locally with `pip3 install -e .`.

- `python3 -m pytest tests` runs the full test suite defined by `pytest.ini`.
- `python3 -m ruff check .` validates lint rules; use `python3 -m ruff check . --fix` for safe auto-fixes.
- `python3 -m ruff format .` applies the repository formatter.
- `python3 -m mypy rl_eng tests` runs static type checks used by pre-commit.
- `python3 -m rl_eng.tic_tac_toe train --epochs 100000 --epsilon 0.75` trains a Tic-Tac-Toe agent and writes a run under `runs/`.
- `python3 scripts/promote_run_to_export.py --run_id <run_id>` promotes a finished run into `artifacts/exports/`.
- `python3 artifacts/apps/tic_tac_toe/launcher.py --run_id <run_id>` launches the GUI for a saved run.

## Coding Style & Naming Conventions
Use Python 3.9+, 4-space indentation, and double quotes. Ruff enforces import ordering, lint rules, and Google-style docstrings; keep line length within 127 characters. 

**Interfaces & Abstract Methods**: When using `@abstractmethod` in classes inheriting from `abc.ABC`, use `pass` instead of `raise NotImplementedError`. Since `ABC` prevents instantiation of incomplete subclasses, the failure occurs at the correct lifecycle point (instantiation) rather than call time.

**Naming**: Prefer explicit, descriptive module and test names, following the existing pattern `tic_tac_toe_td.py` and `test_agent_tic_tac_toe_td.py`. Add type hints where practical; Mypy is configured leniently but still checks function bodies.

## Testing Guidelines
Write tests with `pytest` in `tests/`, naming files `test_<area>.py` and test functions `test_<behavior>()`. Keep tests close to the module they validate and cover environment transitions, rollout behavior, and training-facing interfaces. Run `python3 -m pytest tests -q` before opening a PR; update or add tests whenever agent logic, environment rules, or CLI behavior changes.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Clarify in README: for example tic_tac_toe` and `Move trajectory buffer into data layer`. Follow that style: one-line subject, present tense, focused scope. PRs should include a brief description, linked issue if applicable, exact commands used for verification, and screenshots only for GUI changes under `artifacts/apps/tic_tac_toe/`.
