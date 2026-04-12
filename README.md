# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. It provides extensible abstractions for environments, agents, and deployment-ready game interfaces.

---

## 🏗 Project Architecture
The repository is structured to separate core algorithm logic from interactive interfaces and distribution tools:

*   **`rl_eng/`**: Core library containing environment abstractions and RL agent implementations (TD-Learning, MAB).
*   **`games/`**: Interactive implementations using the core library. Features a Pygame-based Tic-Tac-Toe GUI.
*   **`runs/`**: Experimental tracking. Contains timestamped directories with state-value tables and YAML configurations. These are typically ignored by Git.
*   **`models/stable/`**: The **Model Zoo**. This directory is for "graduated" model artifacts that have been verified and are ready for distribution.

## 🚀 Quick Start

### Installation
```bash
# Clone and install dependencies
git clone https://github.com/bowenlee/rl-eng.git
pip3 install -e .
```

### 1. Training
Train a TD(0) agent for Tic-Tac-Toe. This will create a new directory in `runs/`.
```bash
python3 -m rl_eng.tic_tac_toe train --epochs 100000 --epsilon 0.75
```

### 2. Playing (Experimental)
Launch the Pygame interface using a `run_id` from your local `runs/` folder:
```bash
python3 games/tic_tac_toe/launcher.py --run_id <your_run_id>
```

### 3. Graduating to Stable
Once a model is performing perfectly, graduate it to the **Model Zoo**. This automates versioning and metadata generation:
```bash
python3 scripts/promote_run_to_stable.py --run_id <your_run_id>
```
Artifacts will be stored in `models/stable/<model_name>_vK/`.


## 📦 Distribution
Package your stable models into a standalone macOS `.app` bundle:
```bash
# Point the build script to your stable model path or latest run
./games/tic_tac_toe/build_app.sh --run_id tic_tac_toe_20260412_0014_s42
```

## 🛠 Engineering Standards
*   **Linting/Formatting**: Managed via `ruff` (Google style docstrings).
*   **Configuration**: Type-safe experiment configs using `dataclasses`.
*   **Packaging**: PyInstaller integration for standalone GUI deployment.

## 🗺 Roadmap
- TBD
