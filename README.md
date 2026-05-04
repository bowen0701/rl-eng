# RL-Eng: Reinforcement Learning Research & Engineering

`rl-eng` is a modular framework designed to bridge the gap between RL research prototyping and production-ready engineering. The goal is to provide minimum, extensible abstractions for environments, agents, and deployment-ready game interfaces. This repo is in its early development stage so stay tuned.

---

## 🏗 Project Architecture
The repository is evolving toward a clearer RL systems layout where rollout owns execution, data owns trajectories, and scripts stay thin:

```text
rl-eng/
├── rl_eng/                     # ⭐ core Python package (system)
│   ├── rollout/                # sampling + execution engine (heart)
│   ├── learners/               # PPO / DPO / diffusion optimizers
│   ├── models/                 # Neural Nets (pure functions)
│   ├── envs/                   # interaction backends (gym / text / sim)
│   ├── data/                   # trajectories / buffers / datasets
│   ├── infra/                  # distributed, logging, config, checkpoint
│   └── interfaces/             # contracts between subsystems
│       ├── env.py              # environment state-transition backends
│       ├── model.py            # Neural Nets / value table pure functions
│       ├── learner.py          # optimizer logic (PPO, DPO, etc)
│       └── rollout.py          # sampling + execution orchestration
├── scripts/                    # ⭐ entrypoints (thin orchestration only)
├── experiments/                # configs (YAML / Hydra style)
├── apps/                       # checked-in app launchers and packaging scripts
├── artifacts/
│   ├── exports/
│   └── apps/                   # generated app bundles and build outputs
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

## 📦 Distribution
Package an exported run into a standalone macOS `.app` bundle:
```bash
./apps/tic_tac_toe/build_app.sh --run_id <run_id>
```
Build outputs are written under `artifacts/apps/tic_tac_toe/`.

## 🛠 Engineering Standards
*   **Linting/Formatting**: Managed via `ruff`.
*   **Configuration**: Type-safe experiment configs using `dataclasses`.
*   **Naming**: Prefer explicit, clarified names (e.g., `tests/test_agent_tic_tac_toe_td.py` over `test_agent.py`).
*   **Packaging**: PyInstaller integration for standalone GUI deployment.

## 🗺 Roadmap
- TBD
