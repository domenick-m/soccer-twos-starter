"""
Plots either:
  1. training curves from train_compare.py, or
  2. win-rate comparison against ceia_baseline_agent from the evaluation JSONs
     written by `train_compare.py --evaluate-vs-ceia`.

In `auto` mode, CEIA evaluation results are preferred when available. Otherwise
the script falls back to training-curve mode.

Usage
-----
    python plot_compare.py
    python plot_compare.py --mode ceia --local-dir ./ray_results_compare --out ceia_compare.png
    python plot_compare.py --mode training --local-dir ./ray_results_compare --out training_compare.png
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RUN_SPECS = [
    ("PPO_baseline", "Control"),
    ("PPO_shaped", "Shaped"),
]


def progress_timesteps(csv_path: str) -> int:
    latest_value = -1
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps = row.get("timesteps_total")
            if not timesteps:
                continue
            try:
                latest_value = int(float(timesteps))
            except ValueError:
                continue
    return latest_value


def latest_progress_csv(local_dir: str, run_name: str) -> Optional[str]:
    pattern = os.path.join(local_dir, run_name, "*", "progress.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda path: (progress_timesteps(path), os.path.getmtime(path)))
    return matches[-1]


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def eval_json_path(local_dir: str, run_name: str) -> str:
    return os.path.join(local_dir, "evaluations", f"{run_name}_vs_ceia_baseline.json")


def load_eval_row(local_dir: str, run_name: str, label: str) -> Optional[Dict]:
    path = eval_json_path(local_dir, run_name)
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        payload = json.load(f)

    result = payload.get("result", {})
    policies = result.get("policies", {})
    agent_key = payload.get("agent_module", "compare_checkpoint_agent")
    baseline_key = payload.get("baseline_module", "ceia_baseline_agent")

    agent_stats = policies.get(agent_key)
    baseline_stats = policies.get(baseline_key)
    if agent_stats is None or baseline_stats is None:
        return None

    return {
        "run_name": run_name,
        "label": label,
        "episodes": payload.get("eval_episodes"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "win_rate": agent_stats.get("policy_win_rate"),
        "wins": agent_stats.get("policy_wins"),
        "losses": agent_stats.get("policy_losses"),
        "draws": agent_stats.get("policy_draws"),
        "reward_mean": agent_stats.get("policy_reward_mean"),
        "baseline_win_rate": baseline_stats.get("policy_win_rate"),
    }


def available_eval_rows(local_dir: str) -> List[Dict]:
    rows = []
    for run_name, label in RUN_SPECS:
        row = load_eval_row(local_dir, run_name, label)
        if row is not None:
            rows.append(row)
    return rows


def plot_training_curves(local_dir: str, out_path: str, smooth_window: int):
    missing: List[str] = []
    fig, ax = plt.subplots(figsize=(9, 5))

    for run_name, label in RUN_SPECS:
        csv_path = latest_progress_csv(local_dir, run_name)
        if csv_path is None:
            missing.append(run_name)
            continue

        df = pd.read_csv(csv_path)
        if "timesteps_total" not in df or "episode_reward_mean" not in df:
            print(f"[warn] {csv_path} missing expected columns; skipping.")
            continue

        x = df["timesteps_total"].values
        y = smooth(df["episode_reward_mean"], smooth_window).values
        ax.plot(x, y, label=label, linewidth=2)
        print(f"[ok]   training {run_name}: {len(df)} rows from {csv_path}")

    if missing:
        print(
            "[warn] No progress.csv found for: "
            + ", ".join(missing)
            + f" (expected under {local_dir}/<run>/<trial>/progress.csv)"
        )

    ax.set_xlabel("Environment steps (timesteps_total)")
    ax.set_ylabel("Episode reward mean")
    ax.set_title("SoccerTwos PPO: control vs reward-shaped")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")


def plot_ceia_comparison(local_dir: str, out_path: str):
    rows = available_eval_rows(local_dir)
    if not rows:
        print(
            "[warn] No CEIA evaluation JSONs found under "
            f"{os.path.join(local_dir, 'evaluations')}.\n"
            "Run `train_compare.py --evaluate-vs-ceia` first, or use --mode training."
        )
        return

    labels = [row["label"] for row in rows]
    win_rates = [row["win_rate"] for row in rows]
    losses = [row["losses"] for row in rows]
    draws = [row["draws"] for row in rows]
    wins = [row["wins"] for row in rows]
    episodes = [row["episodes"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, win_rates, color=["#4c78a8", "#f58518"][: len(rows)], width=0.6)

    for bar, win, loss, draw, total in zip(bars, wins, losses, draws, episodes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{win}/{total}\nL {loss} D {draw}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Win rate vs CEIA baseline")
    ax.set_title("SoccerTwos: trained policy vs CEIA baseline")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    for row in rows:
        print(
            "[ok]   ceia "
            f"{row['run_name']}: win_rate={row['win_rate']:.3f}, "
            f"wins={row['wins']}, losses={row['losses']}, draws={row['draws']}, "
            f"episodes={row['episodes']}"
        )
    print(f"\nSaved plot to: {out_path}")


def resolve_mode(local_dir: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "ceia" if available_eval_rows(local_dir) else "training"


def main(local_dir: str, out_path: str, smooth_window: int, mode: str):
    resolved_mode = resolve_mode(local_dir, mode)
    print(f"[mode] {resolved_mode}")

    if resolved_mode == "ceia":
        plot_ceia_comparison(local_dir, out_path)
    else:
        plot_training_curves(local_dir, out_path, smooth_window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=os.path.abspath("./ray_results_compare"))
    parser.add_argument("--out", default=os.path.abspath("./comparison.png"))
    parser.add_argument(
        "--mode",
        choices=["auto", "ceia", "training"],
        default="auto",
        help="Plot CEIA evaluation results, training curves, or choose automatically.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling-mean window over training iterations (used only in training mode).",
    )
    args = parser.parse_args()
    main(args.local_dir, args.out, args.smooth_window, args.mode)
