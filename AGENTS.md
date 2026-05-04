# Repository Guidelines

## Project Structure & Module Organization
`rl_eng/` contains the core library, designed for scalability and future offline RL support:
- `rl_eng/envs/`: Environment logic and interaction backends.
- `rl_eng/agents/`: RL agent implementations (e.g., TD, PPO).
- `rl_eng/rollout/`: Sampling and execution engine.
- `rl_eng/data/`: Data primitives, trajectories, and datasets for online/offline RL.
- `rl_eng/config.py`: Common configurations and type-safe experiment settings.

`experiments/` contains experiment-local code, configs, and runtime artifacts:
- `experiments/<project>/`:
  - `config.yaml`: Top-level configuration for the project.
  - `train.py`: Script for training agents.
  - `eval.py`: Script for evaluating trained agents.
  - `runs/`: Directory for individual run artifacts.
    - `<run_id>` (e.g., `f"{config.name}_{yyyymmdd}_{timestamp}_s{config.seed}_g{git_hash}"`): Dynamically named run directories containing:
      - `config.yml`: Run-specific configuration.
      - `train_metrics.csv`: Metrics collected during training.
      - `eval_metrics.csv`: Metrics collected during evaluation.
      - `train_curve.png`: Visual representation of training progress.
      - `eval_curve.png`: Visual representation of evaluation progress.
      - `checkpoints/`: Saved model checkpoints.

`exports/` contains exported models and associated metadata:
- `exports/<project_v0.x>/`: Versioned export directories.
  - `config.yaml`: Exported configuration.
  - `export_metadata.yaml`: Metadata about the export.
  - `checkpoints/`: Exported model checkpoints.

## Build, Test, and Development Commands
Install locally with `pip3 install -e .`.

- `python3 -m pytest tests` runs the full test suite defined by `pytest.ini`.
- `python3 -m ruff check .` validates lint rules; use `python3 -m ruff check . --fix` for safe auto-fixes.
- `python3 -m ruff format .` applies the repository formatter.
- `python3 -m mypy rl_eng tests` runs static type checks used by pre-commit.
- `python3 -m experiments.tic_tac_toe.train --epochs 100000 --epsilon 0.75` trains a Tic-Tac-Toe agent and writes a run under `experiments/tic_tac_toe/runs/`.
- `python3 scripts/promote_run_to_export.py --run_id <run_id>` promotes a finished run into `exports/`.
- `python3 apps/tic_tac_toe/launcher.py --run_id <run_id>` launches the GUI for a saved run.

## Coding Style & Naming Conventions
Use Python 3.9+, 4-space indentation, and double quotes. Ruff enforces import ordering, lint rules, and Google-style docstrings; keep line length within 127 characters. 

**Interfaces & Abstract Methods**: When using `@abstractmethod` in classes inheriting from `abc.ABC`, use `pass` instead of `raise NotImplementedError`. Since `ABC` prevents instantiation of incomplete subclasses, the failure occurs at the correct lifecycle point (instantiation) rather than call time.

**Naming**: Prefer explicit, descriptive module and test names, following the existing pattern `tic_tac_toe_td.py` and `test_agent_tic_tac_toe_td.py`. Add type hints where practical; Mypy is configured leniently but still checks function bodies.

## Testing Guidelines
Write tests with `pytest` in `tests/`, naming files `test_<area>.py` and test functions `test_<behavior>()`. Keep tests close to the module they validate and cover environment transitions, rollout behavior, and training-facing interfaces. Run `python3 -m pytest tests -q` before opening a PR; update or add tests whenever agent logic, environment rules, or CLI behavior changes.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Clarify in README: for example tic_tac_toe` and `Move trajectory buffer into data layer`. Follow that style: one-line subject, present tense, focused scope. PRs should include a brief description, linked issue if applicable, exact commands used for verification, and screenshots only for GUI changes under `apps/tic_tac_toe/`.
