#!/usr/bin/env python3
"""Plot training and evaluation curves for a saved run."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser(description="Plot Tic-Tac-Toe training and evaluation curves from a run directory.")
    parser.add_argument("--run_id", help="Run identifier under runs/.")
    parser.add_argument("--run_dir", help="Explicit path to the run directory.")
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> str:
    """Resolve the run directory from CLI arguments."""
    if bool(args.run_id) == bool(args.run_dir):
        raise ValueError("Provide exactly one of --run_id or --run_dir")
    return args.run_dir if args.run_dir else os.path.join("runs", args.run_id)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Load a CSV file into memory."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Generate training and evaluation plots for a run."""
    args = parse_args()
    run_dir = resolve_run_dir(args)

    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    eval_path = os.path.join(run_dir, "eval_metrics.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"train_metrics.csv not found at {metrics_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"eval_metrics.csv not found at {eval_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot learning curves. Install it with `pip install matplotlib`.") from exc

    metrics_rows = read_csv_rows(metrics_path)
    eval_rows = read_csv_rows(eval_path)

    _plot_training_curves(plt, metrics_rows, os.path.join(run_dir, "train_curve.png"))
    _plot_eval_curves(plt, eval_rows, os.path.join(run_dir, "eval_curve.png"))


def _plot_training_curves(plt, rows: List[Dict[str, str]], output_path: str) -> None:
    episodes = [int(row["episode"]) for row in rows]
    agent1_win_rate = [float(row["agent1_win_rate"]) for row in rows]
    agent2_win_rate = [float(row["agent2_win_rate"]) for row in rows]
    tie_rate = [float(row["tie_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, agent1_win_rate, label="Agent X win rate", linewidth=2)
    ax.plot(episodes, agent2_win_rate, label="Agent O win rate", linewidth=2)
    ax.plot(episodes, tie_rate, label="Tie rate", linewidth=2, linestyle="--")
    ax.set_title("Training Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_eval_curves(plt, rows: List[Dict[str, str]], output_path: str) -> None:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in rows:
        trained_player = row["trained_player"]
        opponent = row["opponent"]
        grouped[trained_player][opponent]["episode"].append(int(row["episode"]))
        grouped[trained_player][opponent]["win_rate"].append(float(row["win_rate"]))
        grouped[trained_player][opponent]["tie_rate"].append(float(row["tie_rate"]))

    trained_players = sorted(grouped.keys())
    fig, axes = plt.subplots(len(trained_players), 1, figsize=(9, 4 * len(trained_players)), squeeze=False)

    for axis, trained_player in zip(axes.flatten(), trained_players):
        for opponent, series in sorted(grouped[trained_player].items()):
            axis.plot(series["episode"], series["win_rate"], label=f"{opponent} win", linewidth=2)
            axis.plot(series["episode"], series["tie_rate"], label=f"{opponent} tie", linewidth=2, linestyle="--")

        axis.set_title(f"Evaluation Curves: trained {trained_player}")
        axis.set_xlabel("Episode")
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
