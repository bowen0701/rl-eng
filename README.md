# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. The goal is to provide minimum, extensible abstractions for environments, agents, and deployment-ready game interfaces. This repo is in its early development stage so stay tuned.

---

## 🏗 Project Architecture
The repository is organized around a small core package, run-local experiment outputs, and a thin set of scripts and app launchers:

```text
rl-eng/
├── apps/                       # checked-in launcher / packaging scripts
│   └── tic_tac_toe/
├── experiments/                # experiment-local code, configs, and runtime artifacts
│   └── <project>/              # e.g., 'tic_tac_toe'
│       ├── config.yaml         # Top-level configuration for the project
│       ├── train.py            # Script for training agents
│       ├── eval.py             # Script for evaluating trained agents
│       └── runs/               # Directory for individual run artifacts
│           └── <run_id>        # e.g., f"{config.name}_{yyyymmdd}_{timestamp}_s{config.seed}_g{git_hash}"
│               ├── config.yml
│               ├── train_metrics.csv
│               ├── eval_metrics.csv
│               ├── train_curve.png
│               ├── eval_curve.png
│               └── checkpoints/
├── rl_eng/                     # core Python package
│   ├── agents/
│   ├── data/
│   ├── envs/
│   ├── evaluation/
│   ├── learners/
│   ├── models/
│   └── rollout/
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
                │ experiments  │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   rollout    │  training loop + metrics
                └──────┬───────┘
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        envs       evaluation      learners
          ↓            ↓
        data         models
```

## 🚀 Quick Start

### Installation
```bash
# Clone and install dependencies
git clone https://github.com/bowenlee/rl-eng.git
cd rl-eng
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```
```

### 1. Training & Testing
For example: Train a TD(0) agent for Tic-Tac-Toe. This will create a new directory under `experiments/<project>/runs/`.
```bash
python3 -m experiments.<project>.train --epochs 100000 --epsilon 0.75
```
Training writes per-run outputs under `experiments/tic_tac_toe/runs/<run_id>/`, including `config.yml`, `train_metrics.csv`, `eval_metrics.csv`, `train_curve.png`, `eval_curve.png`, and `checkpoints/`.
Run the automated test suite:
```bash
python3 -m pytest tests
```

### 2. Playing (Experimental)
Launch the Pygame interface using a `run_id` from your local `experiments/<project>/runs/` folder:
```bash
python3 apps/tic_tac_toe/launcher.py --run_id <your_run_id>
```

### 3. Promoting to Exports
Once a run is ready for reuse, promote it into the exports bucket. This automates versioning and metadata generation:
```bash
python3 scripts/promote_run_to_export.py --run_id <your_run_id> --version 0.1
```
Artifacts will be stored in `exports/<project_v0.x>/`.

### 4. Plotting Learning Curves
Generate separate training and evaluation plots from a saved run:
```bash
python3 scripts/plot_learning_curves.py --run_id <your_run_id>
```
This reads `experiments/<project>/runs/<run_id>/train_metrics.csv` and `experiments/<project>/runs/<run_id>/eval_metrics.csv` and writes `train_curve.png` and `eval_curve.png` back into the run directory.

## 📦 Distribution
Package an exported run into a standalone macOS `.app` bundle:
```bash
./apps/tic_tac_toe/build_app.sh --run_id <run_id>
```
Build outputs are written under `artifacts/apps/tic_tac_toe/`, while the app source lives under `apps/tic_tac_toe/`.

## 🛠 Engineering Standards
*   **Linting/Formatting**: Managed via `ruff`.
*   **Configuration**: Type-safe experiment configs using `dataclasses`.
*   **Naming**: Prefer explicit, clarified names (e.g., `tests/test_agent_tic_tac_toe_td.py` over `test_agent.py`).
*   **Packaging**: PyInstaller integration for standalone GUI deployment.

## 🗺 Roadmap
- TBD
