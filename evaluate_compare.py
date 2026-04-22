"""
Post hoc evaluation helper for existing comparison runs.

Given existing PPO_baseline / PPO_shaped trial directories, this script:
  1. finds the latest checkpoint in each trial directory,
  2. evaluates that checkpoint against `ceia_baseline_agent`,
  3. writes JSON summaries under `./ray_results_compare/evaluations/`.

After running this script, use:
    python plot_compare.py --mode ceia --local-dir ./ray_results_compare

Usage
-----
    python evaluate_compare.py
    python evaluate_compare.py --baseline-trial-dir ./ray_results_compare/PPO_baseline/<trial>
    python evaluate_compare.py --shaped-trial-dir ./ray_results_compare/PPO_shaped/<trial>
    python evaluate_compare.py --eval-episodes 20 --base-port 52039
"""

import argparse
import os
from typing import Optional

from train_compare import evaluate_against_ceia, find_reusable_run, latest_checkpoint_path


RUN_SPECS = [
    ("PPO_baseline", "baseline_trial_dir"),
    ("PPO_shaped", "shaped_trial_dir"),
]


def resolve_trial_dir(
    local_dir: str,
    run_name: str,
    explicit_trial_dir: Optional[str],
) -> Optional[str]:
    if explicit_trial_dir:
        return os.path.abspath(explicit_trial_dir)

    reusable = find_reusable_run(run_name, local_dir, min_timesteps=0)
    if reusable is None:
        return None
    return reusable["trial_dir"]


def main(
    local_dir: str,
    baseline_trial_dir: Optional[str],
    shaped_trial_dir: Optional[str],
    eval_episodes: int,
    base_port: Optional[int],
):
    trial_dirs = {
        "PPO_baseline": resolve_trial_dir(local_dir, "PPO_baseline", baseline_trial_dir),
        "PPO_shaped": resolve_trial_dir(local_dir, "PPO_shaped", shaped_trial_dir),
    }

    found_any = False
    for run_name, trial_dir in trial_dirs.items():
        if trial_dir is None:
            print(f"[skip] {run_name}: no trial directory found")
            continue

        checkpoint_path = latest_checkpoint_path(trial_dir)
        print(f"[eval] {run_name}")
        print(f"[trial] {trial_dir}")
        print(f"[checkpoint] {checkpoint_path}")

        evaluate_against_ceia(
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            local_dir=local_dir,
            eval_episodes=eval_episodes,
            eval_base_port=base_port,
        )
        found_any = True

    if not found_any:
        print(
            "[warn] No baseline/shaped trial dirs were found. "
            "Pass --baseline-trial-dir and/or --shaped-trial-dir explicitly."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=os.path.abspath("./ray_results_compare"))
    parser.add_argument(
        "--baseline-trial-dir",
        default=None,
        help="Optional explicit PPO_baseline trial directory to evaluate.",
    )
    parser.add_argument(
        "--shaped-trial-dir",
        default=None,
        help="Optional explicit PPO_shaped trial directory to evaluate.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of episodes to play against ceia_baseline_agent.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=52039,
        help="Base port for post hoc CEIA evaluation matches.",
    )
    args = parser.parse_args()
    main(
        local_dir=os.path.abspath(args.local_dir),
        baseline_trial_dir=args.baseline_trial_dir,
        shaped_trial_dir=args.shaped_trial_dir,
        eval_episodes=args.eval_episodes,
        base_port=args.base_port,
    )
