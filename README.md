# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. The goal is to provide minimum, extensible abstractions for environments, agents, and deployment-ready game interfaces. This repo is in its early development stage so stay tuned.

---

## 🏗 Project Architecture
The repository is organized around a small core package, run-local experiment outputs, and a thin set of scripts and app launchers:

```text
rl-eng/
├── apps/                       # checked-in launcher / packaging scripts
│   └── tic_tac_toe/
├── rl_eng/                     # core Python package
│   ├── agents/
│   ├── data/
│   ├── envs/
│   ├── evaluation/
│   ├── learners/
│   ├── models/
│   ├── rollout/
│   └── tic_tac_toe.py          # training / play CLI
├── scripts/                    # utility scripts (promotion, plotting)
├── artifacts/
│   ├── exports/                # promoted model exports
│   └── apps/                   # generated app bundles and build outputs
├── runs/                       # run-local configs, tables, metrics, curves
├── tests/
├── pyproject.toml
└── README.md
```

### Mental Model
```text
                ┌──────────────┐
                │   scripts    │
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
pip3 install -e .
```

### 1. Training & Testing
For example: Train a TD(0) agent for Tic-Tac-Toe. This will create a new directory in `runs/`.
```bash
python3 -m rl_eng.tic_tac_toe train --epochs 100000 --epsilon 0.75
```
Training writes per-run outputs under `runs/<run_id>/`, including `config.yaml`, `state_values_*.json`, `metrics.csv`, `eval.csv`, and curve images.
Run the automated test suite:
```bash
python3 -m pytest tests
```

### 2. Playing (Experimental)
Launch the Pygame interface using a `run_id` from your local `runs/` folder:
```bash
python3 apps/tic_tac_toe/launcher.py --run_id <your_run_id>
```

### 3. Promoting to Exports
Once a run is ready for reuse, promote it into the exports bucket. This automates versioning and metadata generation:
```bash
python3 scripts/promote_run_to_export.py --run_id <your_run_id>
```
Artifacts will be stored in `artifacts/exports/<model_name>_vK/`.

### 4. Plotting Learning Curves
Generate separate training and evaluation plots from a saved run:
```bash
python3 scripts/plot_learning_curves.py --run_id <your_run_id>
```
This reads `runs/<run_id>/metrics.csv` and `runs/<run_id>/eval.csv` and writes `training_curves.png` and `evaluation_curves.png` back into the run directory.

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
