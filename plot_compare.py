"""
Overlays learning curves for the two runs produced by train_compare.py.

Reads the per-iteration `progress.csv` that Ray Tune writes inside each run's
trial directory and plots `episode_reward_mean` against `timesteps_total`.
Saves the figure next to the ray_results dir.

Note: the `shaped` run reports reward on the shaped scale (goal events
amplified by 3x by default), so absolute values are not directly comparable.
Curve *shape* - slope, plateau, time-to-positive-reward - is what matters.

Usage
-----
    python plot_compare.py
    python plot_compare.py --local-dir ./ray_results_compare --out comparison.png
"""

import argparse
import glob
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


RUN_SPECS = [
    ("PPO_baseline", "Baseline (raw env reward)"),
    ("PPO_shaped", "Shaped (goal x3 + ball-distance)"),
]


def latest_progress_csv(local_dir: str, run_name: str) -> Optional[str]:
    """Find the most recent trial's progress.csv under `local_dir/run_name`."""
    pattern = os.path.join(local_dir, run_name, "*", "progress.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def main(local_dir: str, out_path: str, smooth_window: int):
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
        print(f"[ok]   {run_name}: {len(df)} rows from {csv_path}")

    if missing:
        print(
            "[warn] No progress.csv found for: "
            + ", ".join(missing)
            + f" (expected under {local_dir}/<run>/<trial>/progress.csv)"
        )

    ax.set_xlabel("Environment steps (timesteps_total)")
    ax.set_ylabel("Episode reward mean")
    ax.set_title("SoccerTwos PPO: baseline vs reward-shaped")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=os.path.abspath("./ray_results_compare"))
    parser.add_argument("--out", default=os.path.abspath("./comparison.png"))
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling-mean window over training iterations (1 = no smoothing).",
    )
    args = parser.parse_args()
    main(args.local_dir, args.out, args.smooth_window)
