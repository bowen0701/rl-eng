# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. The goal is to provide minimum, extensible abstractions for environments, agents, and deployment-ready game interfaces. This repo is in its early development stage so stay tuned.

---

## 🏗 Project Architecture
The repository is organized around a small generic core package (`rl_eng/`), per-project code and artifacts (`projects/`), and a thin set of scripts and app launchers:

```text
rl-eng/
├── apps/                       # checked-in launcher / packaging scripts
│   └── tic_tac_toe/
├── projects/                   # one directory per RL project
│   └── <project>/              # e.g., 'tic_tac_toe'
│       ├── env/                # project-specific environment
│       ├── agent.py            # project-specific agent
│       ├── evaluation.py       # project-specific evaluation helpers
│       ├── rollout.py          # project-specific training loop
│       ├── config.yaml         # default hyperparameters
│       ├── train.py            # training entry point
│       ├── eval.py             # evaluation / play entry point
│       ├── notebooks/          # project-specific notebooks
│       └── runs/               # runtime artifacts (not committed)
│           └── <run_id>        # e.g., f"{env}_{yyyymmdd}_{hhmm}_s{seed}_g{git_hash}"
│               ├── config.yml
│               ├── train_metrics.csv
│               ├── eval_metrics.csv
│               ├── train_curve.png
│               ├── eval_curve.png
│               └── checkpoints/
├── rl_eng/                     # generic core library (interfaces + algorithms)
│   ├── interfaces/             # abstract base classes (Env, Learner, Model, Rollout)
│   ├── data/                   # Trajectory
│   ├── learners/               # TDLearner and future generic learners
│   ├── models/                 # StateValueTable and future generic models
│   └── config.py               # BaseConfig / TrainingConfig dataclasses
├── scripts/                    # utility scripts (promotion, plotting)
├── exports/                    # promoted model exports
│   └── <project_v0.x>/         # e.g., 'tic_tac_toe_v0.1'
│       ├── config.yaml
│       ├── export_metadata.yaml
│       └── checkpoints/
├── tests/
├── pyproject.toml
└── README.md
```

### Mental Model
```text
                ┌──────────────┐
                │   projects   │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   rollout    │  training loop + metrics
                └──────┬───────┘
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        env        evaluation      rl_eng/learners
          ↓            ↓
        rl_eng/    rl_eng/models
        data
```

## 🚀 Quick Start

### Installation

**Remote server** (conda base env is active by default):
```bash
git clone https://github.com/bowenlee/rl-eng.git && cd rl-eng && make install
```

**Local development** (activate a venv or conda env first):
```bash
git clone https://github.com/bowenlee/rl-eng.git && cd rl-eng
python3 -m venv .venv && source .venv/bin/activate
make install
```

### Developer Workflow

| Target           | Description                         |
|------------------|-------------------------------------|
| `make install`   | Install package in editable mode    |
| `make test`      | Run test suite                      |
| `make lint`      | Lint and auto-fix with ruff         |
| `make format`    | Format with ruff                    |
| `make typecheck` | Type-check core library with mypy   |

### 1. Train
Each project exposes a `train.py` entry point. Run it as a module from the repo root:
```bash
python3 -m projects.<project>.train [--epochs N] [--step_size LR] [--epsilon E] [--seed S]
```
Outputs are written to `projects/<project>/runs/<run_id>/` and include `config.yml`, `train_metrics.csv`, `eval_metrics.csv`, and `checkpoints/`.

### 2. Evaluate / Play
Each project exposes an `eval.py` entry point. Load a trained run and play:
```bash
python3 -m projects.<project>.eval play --run_id <run_id>
```

### 3. Plot Learning Curves
Generate training and evaluation plots from any saved run:
```bash
python3 scripts/plot_learning_curves.py --run_id <run_id>
```
Writes `train_curve.png` and `eval_curve.png` into `projects/<project>/runs/<run_id>/`.

### 4. Promote to Exports
Promote a finished run into the versioned exports bucket:
```bash
python3 scripts/promote_run_to_export.py --run_id <run_id> [--version 0.1]
```
Artifacts are stored under `exports/<project>_v<X.Y>/`.

## 📦 Distribution
Package an exported run into a standalone macOS `.app` bundle (per project):
```bash
./apps/<project>/build_app.sh --run_id <run_id>
```
Build outputs are written under `artifacts/apps/<project>/`, while the app source lives under `apps/<project>/`.

## 🛠 Engineering Standards
*   **Linting/Formatting**: Managed via `ruff`.
*   **Configuration**: Type-safe experiment configs using `dataclasses`.
*   **Naming**: Prefer explicit, clarified names (e.g., `tests/test_agent_tic_tac_toe_td.py` over `test_agent.py`).
*   **Packaging**: PyInstaller integration for standalone GUI deployment.

## 🗺 Roadmap
- TBD
