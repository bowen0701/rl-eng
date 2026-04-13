# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. The goal is to provide minimum, extensible abstractions for environments, agents, and deployment-ready game interfaces. This repo is in its early development stage so stay tuned.

---

## 🏗 Project Architecture
The repository is structured to separate core algorithm logic from interactive interfaces and infrastructure tools:

*   **`rl_eng/`**: Core library containing environment abstractions (`envs/`) and agent implementations (`agents/`).
*   **`games/`**: Interactive implementations using the core library. Features a Pygame-based Tic-Tac-Toe GUI.
*   **`runs/`**: Local experimental tracking. Contains timestamped directories with state-value tables and YAML configurations. **Note: This directory is ignored by Git to prevent committing transient training artifacts.**
*   **`models/stable/`**: The **Model Zoo**. Verified artifacts ready for distribution.
*   **`scripts/`**: Automation tools for model graduation and infrastructure management.
*   **`tests/`**: Automated verification for core infrastructure and RL logic.

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
Run the automated test suite:
```bash
python3 -m pytest tests
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
./games/tic_tac_toe/build_app.sh --run_id <run_id>
```

## 🛠 Engineering Standards
*   **Linting/Formatting**: Managed via `ruff`.
*   **Configuration**: Type-safe experiment configs using `dataclasses`.
*   **Naming**: Prefer explicit, clarified names (e.g., `tests/test_agent_tic_tac_toe_td.py` over `test_agent.py`).
*   **Packaging**: PyInstaller integration for standalone GUI deployment.

## 🗺 Roadmap
- TBD
