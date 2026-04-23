"""
Backfill or refresh CEIA evaluation metrics for an existing reward sweep.

This is useful while a sweep is still running: it walks the sweep directory,
finds the latest checkpoint for each configured reward-shaping run, evaluates
that checkpoint against `ceia_baseline_agent`, and writes/updates the shared
`reward_sweep_summary.csv`.

Usage
-----
    python evaluate_reward_sweep.py --local-dir ./ray_results_reward_sweep_v2
    python evaluate_reward_sweep.py --local-dir ./ray_results_reward_sweep_v2 --summary-only
"""

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List

import ray

from train_compare import evaluate_against_ceia
from train_reward_sweep import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_LOCAL_DIR,
    build_summary_row_from_metadata,
    find_trial_dir_for_run_name,
    infer_local_trial_status,
    latest_checkpoint_path_or_none,
    load_cached_eval_payload,
    load_latest_result,
    write_summary_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=None,
        help="Optional limit on how many planned trials to inspect.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not run new CEIA matches; only summarize current trial state and cached eval JSONs.",
    )
    parser.add_argument(
        "--terminated-only",
        action="store_true",
        help="Only run fresh CEIA evaluation for trials whose latest result is marked done.",
    )
    return parser.parse_args()


def load_trial_specs(local_dir: str) -> List[Dict[str, Any]]:
    plan_path = os.path.join(local_dir, "reward_sweep_plan.json")
    if not os.path.isfile(plan_path):
        raise FileNotFoundError(f"Missing sweep plan: {plan_path}")

    with open(plan_path) as f:
        payload = json.load(f)
    return payload.get("trials", [])


def ensure_eval_ray_initialized():
    if ray.is_initialized():
        return

    try:
        ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
        print("[ray] connected to existing Ray cluster for evaluation")
        return
    except Exception:
        pass

    ray_temp_dir = os.path.join(
        tempfile.gettempdir(),
        f"ray_reward_sweep_eval_{os.getpid()}",
    )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=ray_temp_dir,
        num_cpus=1,
        num_gpus=0,
    )
    print(f"[ray] started isolated evaluation cluster in {ray_temp_dir}")


def main():
    args = parse_args()
    local_dir = os.path.abspath(args.local_dir)
    experiment_dir = os.path.join(local_dir, args.experiment_name)
    summary_csv_path = os.path.join(local_dir, "reward_sweep_summary.csv")

    trial_specs = load_trial_specs(local_dir)
    if args.trial_limit is not None:
        trial_specs = trial_specs[: args.trial_limit]

    if not trial_specs:
        raise ValueError(f"No trials found in {os.path.join(local_dir, 'reward_sweep_plan.json')}")

    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")

    rows_by_name = {
        trial_spec["run_name"]: build_summary_row_from_metadata(
            trial_spec=trial_spec,
            local_dir=local_dir,
            status="PENDING",
        )
        for trial_spec in trial_specs
    }

    def ordered_rows():
        return [rows_by_name[trial_spec["run_name"]] for trial_spec in trial_specs]

    write_summary_csv(ordered_rows(), summary_csv_path)

    if not args.summary_only:
        ensure_eval_ray_initialized()

    evaluated_count = 0
    for trial_spec in trial_specs:
        run_name = trial_spec["run_name"]
        trial_dir = find_trial_dir_for_run_name(experiment_dir, run_name)
        last_result = load_latest_result(trial_dir)
        status = infer_local_trial_status(trial_dir, last_result)
        checkpoint_path = latest_checkpoint_path_or_none(trial_dir)
        error_file = None
        if trial_dir is not None:
            candidate_error_path = os.path.join(trial_dir, "error.txt")
            if os.path.isfile(candidate_error_path):
                error_file = candidate_error_path

        eval_payload = load_cached_eval_payload(
            local_dir=local_dir,
            run_name=run_name,
            checkpoint_path=checkpoint_path,
            eval_episodes=args.eval_episodes,
        )

        should_eval = (
            not args.summary_only
            and checkpoint_path is not None
            and (not args.terminated_only or status == "TERMINATED")
            and eval_payload is None
        )
        if should_eval:
            print(
                f"\n=== Evaluating {run_name} vs ceia_baseline_agent "
                f"(status={status}, base_port={trial_spec['eval_base_port']}) ==="
            )
            eval_payload = evaluate_against_ceia(
                checkpoint_path=checkpoint_path,
                run_name=run_name,
                local_dir=local_dir,
                eval_episodes=args.eval_episodes,
                eval_base_port=trial_spec["eval_base_port"],
            )
            evaluated_count += 1

        rows_by_name[run_name] = build_summary_row_from_metadata(
            trial_spec=trial_spec,
            local_dir=local_dir,
            status=status,
            trial_dir=trial_dir,
            last_result=last_result,
            checkpoint_path=checkpoint_path,
            eval_payload=eval_payload,
            error_file=error_file,
        )
        write_summary_csv(ordered_rows(), summary_csv_path)

    status_counts: Dict[str, int] = {}
    for row in ordered_rows():
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(
        "\nDone.\n"
        f"[summary] {summary_csv_path}\n"
        f"[trials]   {status_counts}\n"
        f"[evals]    newly_run={evaluated_count}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if ray.is_initialized():
            ray.shutdown()
