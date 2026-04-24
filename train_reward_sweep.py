"""
Parallel reward-weight sweep for SoccerTwos PPO.

This script intentionally uses a single local Ray cluster for the whole sweep.
That avoids the Redis/head-port conflicts you get when launching many separate
Ray trainers in parallel. Each Tune trial still receives its own SoccerTwos
`base_port` range so the Unity workers do not collide with one another.

Workflow:
  1. Build a set of reward-shaping configs from `reward_weight_sweep.json`.
  2. Train all configs as separate Tune trials on one Ray cluster.
  3. Evaluate each finished checkpoint against `ceia_baseline_agent`.
  4. Write a summary CSV containing reward weights and CEIA results.

Usage
-----
    python train_reward_sweep.py --timesteps 500000 --parallel-trials 2
    python train_reward_sweep.py --trial-limit 3 --dry-run
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import socket
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from itertools import product
from typing import Any, Dict, List, Mapping, Optional

import ray
from ray import tune
from ray.tune.callback import Callback
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from reward_shaping import create_rllib_env_shaped, get_default_reward_shaping_config
from train_compare import (
    CEIA_AGENT_MODULE,
    CHECKPOINT_AGENT_MODULE,
    NUM_ENVS_PER_WORKER,
    evaluate_against_ceia,
    latest_checkpoint_path,
    latest_recorded_timesteps,
    normalize_for_json,
    open_probe_env,
)


DEFAULT_EXPERIMENT_NAME = "PPO_reward_weight_sweep"
DEFAULT_LOCAL_DIR = os.path.abspath("./ray_results_reward_sweep")
DEFAULT_SPEC_PATH = os.path.join(
    os.path.dirname(__file__),
    "reward_weight_sweep.json",
)
ENV_NAME = "SoccerRewardSweep"
PATH_LABELS = {
    "ball_velocity_to_goal.weight": "ball_vel_goal_w",
    "goal_distance.weight": "goal_dist_w",
    "goal_velocity.weight": "goal_vel_w",
    "teammate_spacing.weight": "spacing_w",
    "possession_balance.weight": "poss_w",
    "possession_balance.pass_bonus": "pass_bonus",
    "ball_proximity.weight": "ball_prox_w",
}


def default_policy_mapping_fn(_):
    return "default"


def parse_nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "Expected a non-negative value, e.g. 0, 0.5, or 1."
        )
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--num-gpus",
        type=parse_nonnegative_float,
        default=1.0,
        help=(
            "GPUs per PPO trial. Use fractional values (e.g. 0.5, 0.25) "
            "to pack multiple trials onto one GPU."
        ),
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help="Target number of PPO trials to run concurrently on the shared Ray cluster.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Optional CPU cap for the shared Ray cluster. Defaults to a cap derived from --parallel-trials.",
    )
    parser.add_argument(
        "--ray-num-gpus",
        type=float,
        default=None,
        help="Optional GPU cap for the shared Ray cluster.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help=(
            "Optional comma-separated physical GPU ids to expose to the sweep via "
            "CUDA_VISIBLE_DEVICES, e.g. '2,3'. Useful on shared machines."
        ),
    )
    parser.add_argument(
        "--base-port-start",
        type=int,
        default=50039,
        help="First SoccerTwos base port used for training trials.",
    )
    parser.add_argument(
        "--eval-base-port-start",
        type=int,
        default=None,
        help="First SoccerTwos base port used for CEIA evaluation. Defaults to a disjoint range after training ports.",
    )
    parser.add_argument(
        "--probe-base-port",
        type=int,
        default=None,
        help="Optional dedicated base port for the temporary probe env used to fetch spaces.",
    )
    parser.add_argument(
        "--port-stride",
        type=int,
        default=128,
        help="Port range reserved per trial. Keep this comfortably above num_workers*num_envs_per_worker.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-parallelism",
        type=int,
        default=1,
        help=(
            "Number of concurrent CEIA evaluations after training. "
            "Set >1 to parallelize post-training evaluation."
        ),
    )
    parser.add_argument(
        "--sweep-spec",
        default=DEFAULT_SPEC_PATH,
        help="JSON file describing the reward-weight sweep.",
    )
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Tune checkpoint frequency in training iterations.",
    )
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=None,
        help="Optional limit for the number of sweep configs to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the sweep plan and exit without launching training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the CEIA evaluation phase and only train the sweep.",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_ids_arg: Optional[str]) -> Optional[List[int]]:
    if gpu_ids_arg is None:
        return None

    parsed_ids: List[int] = []
    for raw_part in gpu_ids_arg.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            gpu_id = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU id {part!r} in --gpu-ids. Expected comma-separated integers."
            ) from exc
        if gpu_id < 0:
            raise ValueError("--gpu-ids cannot contain negative GPU ids.")
        parsed_ids.append(gpu_id)

    if not parsed_ids:
        raise ValueError("--gpu-ids was provided but no valid GPU ids were found.")
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError("--gpu-ids contains duplicates; provide each GPU id once.")
    return parsed_ids


def configure_visible_gpus(args):
    selected_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if selected_gpu_ids is None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in selected_gpu_ids)
    if args.ray_num_gpus is None:
        args.ray_num_gpus = float(len(selected_gpu_ids))

    if args.num_gpus > len(selected_gpu_ids):
        raise ValueError(
            f"--num-gpus={args.num_gpus} requests more GPUs per trial than "
            f"the {len(selected_gpu_ids)} GPU(s) exposed by --gpu-ids={args.gpu_ids}."
        )

    print(
        "[gpu] Restricting sweep to physical GPU ids "
        f"{selected_gpu_ids} via CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )


def validate_resource_args(args):
    if args.ray_num_gpus is not None and args.num_gpus > args.ray_num_gpus:
        raise ValueError(
            f"--num-gpus={args.num_gpus} exceeds --ray-num-gpus={args.ray_num_gpus}. "
            "Per-trial GPU count cannot exceed the shared Ray cluster GPU cap."
        )


def default_sweep_spec() -> Dict[str, Any]:
    return {
        "sweep_mode": "one_at_a_time",
        "include_goal_event_only": True,
        "include_default": True,
        "joint_dense_weight_scales": [0.5, 1.5, 2.0],
        "weight_sweeps": {
            "ball_velocity_to_goal.weight": [0.0, 0.01, 0.04],
            "goal_distance.weight": [0.0, 0.005, 0.02],
            "goal_velocity.weight": [0.0, 0.005, 0.02],
            "teammate_spacing.weight": [0.0, 0.005, 0.02],
            "possession_balance.weight": [0.0, 0.005, 0.02],
            "possession_balance.pass_bonus": [0.0, 0.01, 0.04],
        },
    }


def load_sweep_spec(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        print(f"[warn] Sweep spec not found at {path}; using built-in defaults.")
        return default_sweep_spec()

    with open(path) as f:
        return json.load(f)


def get_nested_value(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    value = mapping
    for part in dotted_path.split("."):
        value = value[part]
    return value


def set_nested_value(mapping: Dict[str, Any], dotted_path: str, value: Any):
    target = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def flatten_dict(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def format_value_label(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "trial"


def make_weight_sweep_label(path: str, value: Any) -> str:
    base = PATH_LABELS.get(path, path.replace(".", "_"))
    return f"{base}_{format_value_label(value)}"


def scale_dense_weights(config: Dict[str, Any], dotted_paths: List[str], scale: float):
    for path in dotted_paths:
        current = get_nested_value(config, path)
        if isinstance(current, (int, float)):
            set_nested_value(config, path, round(float(current) * scale, 8))


def zero_dense_weights(config: Dict[str, Any], dotted_paths: List[str]):
    for path in dotted_paths:
        current = get_nested_value(config, path)
        if isinstance(current, (int, float)):
            set_nested_value(config, path, 0.0)


def build_trial_specs(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    defaults = get_default_reward_shaping_config()
    weight_sweeps = spec.get("weight_sweeps", {})
    if not weight_sweeps:
        raise ValueError("Sweep spec must define at least one entry in `weight_sweeps`.")

    sweep_mode = str(spec.get("sweep_mode", "one_at_a_time")).strip().lower()
    valid_modes = {"one_at_a_time", "cartesian"}
    if sweep_mode not in valid_modes:
        raise ValueError(
            f"Unsupported sweep_mode {sweep_mode!r}. Expected one of: {sorted(valid_modes)}"
        )

    dense_paths = list(weight_sweeps.keys())
    trial_specs: List[Dict[str, Any]] = []

    include_reference_trials = (
        sweep_mode != "cartesian"
        or spec.get("include_reference_trials_in_cartesian", False)
    )
    if include_reference_trials:
        if spec.get("include_goal_event_only", True):
            config = copy.deepcopy(defaults)
            zero_dense_weights(config, dense_paths)
            trial_specs.append(
                {
                    "label": "goal_event_only",
                    "reward_shaping": config,
                    "source": "goal_event_only",
                }
            )

        if spec.get("include_default", True):
            trial_specs.append(
                {
                    "label": "default",
                    "reward_shaping": copy.deepcopy(defaults),
                    "source": "default",
                }
            )

        for scale in spec.get("joint_dense_weight_scales", []):
            config = copy.deepcopy(defaults)
            scale_dense_weights(config, dense_paths, float(scale))
            trial_specs.append(
                {
                    "label": f"dense_scale_{format_value_label(scale)}",
                    "reward_shaping": config,
                    "source": "joint_dense_scale",
                }
            )

    if sweep_mode == "cartesian":
        sweep_paths = list(weight_sweeps.keys())
        sweep_values = []
        for path in sweep_paths:
            values = weight_sweeps[path]
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"weight_sweeps[{path!r}] must be a non-empty list for cartesian mode."
                )
            sweep_values.append(values)

        for combo_index, combo in enumerate(product(*sweep_values)):
            config = copy.deepcopy(defaults)
            for path, value in zip(sweep_paths, combo):
                set_nested_value(config, path, value)
            trial_specs.append(
                {
                    "label": f"grid_{combo_index:04d}",
                    "reward_shaping": config,
                    "source": "cartesian_grid",
                }
            )
    else:
        for path, values in weight_sweeps.items():
            for value in values:
                config = copy.deepcopy(defaults)
                set_nested_value(config, path, value)
                trial_specs.append(
                    {
                        "label": make_weight_sweep_label(path, value),
                        "reward_shaping": config,
                        "source": path,
                    }
                )

    deduped_specs: List[Dict[str, Any]] = []
    seen_fingerprints = set()
    for trial_spec in trial_specs:
        fingerprint = json.dumps(
            normalize_for_json(trial_spec["reward_shaping"]),
            sort_keys=True,
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        deduped_specs.append(trial_spec)

    return deduped_specs


def assign_ports(
    trial_specs: List[Dict[str, Any]],
    base_port_start: int,
    port_stride: int,
    eval_base_port_start: Optional[int],
    probe_base_port: Optional[int],
) -> int:
    if not trial_specs:
        raise ValueError("No sweep trials were generated.")

    def port_is_free(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def port_block_is_free(base_port: int, block_size: int) -> bool:
        if base_port < 1 or base_port + block_size - 1 > 65535:
            return False
        return all(port_is_free(base_port + offset) for offset in range(block_size))

    def find_free_port_block(start_port: int, block_size: int) -> int:
        candidate = start_port
        while candidate + block_size - 1 <= 65535:
            if port_block_is_free(candidate, block_size):
                return candidate
            candidate += block_size
        raise RuntimeError(
            f"Could not find a free port block of size {block_size} starting from {start_port}."
        )

    next_train_base_port = find_free_port_block(base_port_start, port_stride)

    for index, trial_spec in enumerate(trial_specs):
        trial_spec["index"] = index
        trial_spec["run_name"] = f"{index:03d}_{sanitize_name(trial_spec['label'])}"
        trial_spec["train_base_port"] = next_train_base_port
        next_train_base_port = find_free_port_block(
            next_train_base_port + port_stride,
            port_stride,
        )

    if eval_base_port_start is None:
        eval_base_port_start = next_train_base_port + port_stride * 2

    next_eval_base_port = find_free_port_block(eval_base_port_start, port_stride)
    for trial_spec in trial_specs:
        trial_spec["eval_base_port"] = next_eval_base_port
        next_eval_base_port = find_free_port_block(
            next_eval_base_port + port_stride,
            port_stride,
        )

    if probe_base_port is None:
        probe_base_port = next_eval_base_port + port_stride * 2
    probe_base_port = find_free_port_block(probe_base_port, port_stride)

    return probe_base_port


def trial_run_name(trial):
    return trial.config["env_config"]["_sweep_meta"]["run_name"]


def build_trial_dirname(trial):
    return trial.config["env_config"]["_sweep_meta"]["run_name"]


def build_env_config_from_trial_spec(trial_spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player,
        "base_port": trial_spec["train_base_port"],
        "reward_shaping": trial_spec["reward_shaping"],
        "_sweep_meta": {
            "index": trial_spec["index"],
            "label": trial_spec["label"],
            "run_name": trial_spec["run_name"],
            "source": trial_spec["source"],
            "train_base_port": trial_spec["train_base_port"],
            "eval_base_port": trial_spec["eval_base_port"],
        },
    }


def small_timestep_ppo_overrides(
    timesteps: int,
    num_workers: int,
) -> Dict[str, int]:
    """Keep smoke runs fast by shrinking PPO batch sizes when timesteps are tiny.

    RLlib PPO defaults to train_batch_size=4000. For very small stop targets
    (e.g. --timesteps 500), that default can make the first training iteration
    take much longer than expected because Tune only checks stop criteria
    between iterations. For normal training runs (timesteps >= 4000), we keep
    RLlib defaults unchanged.
    """
    if timesteps >= 4000:
        return {}

    effective_workers = max(1, num_workers)
    effective_env_runners = effective_workers * max(1, NUM_ENVS_PER_WORKER)
    train_batch_size = max(effective_env_runners, max(1, timesteps))
    rollout_fragment_length = max(
        1,
        int(math.ceil(train_batch_size / float(effective_env_runners))),
    )
    sgd_minibatch_size = min(128, train_batch_size)
    return {
        "train_batch_size": train_batch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "sgd_minibatch_size": sgd_minibatch_size,
    }


def build_tune_config(
    obs_space,
    act_space,
    num_workers: int,
    num_gpus: float,
    trial_specs,
    timesteps: int,
):
    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",
        "multiagent": {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": default_policy_mapping_fn,
            "policies_to_train": ["default"],
        },
        "env": ENV_NAME,
        "env_config": tune.grid_search(
            [build_env_config_from_trial_spec(trial_spec) for trial_spec in trial_specs]
        ),
    }
    ppo_overrides = small_timestep_ppo_overrides(
        timesteps=timesteps,
        num_workers=num_workers,
    )
    if ppo_overrides:
        print(
            "[ppo] small-timestep mode: "
            f"train_batch_size={ppo_overrides['train_batch_size']}, "
            f"rollout_fragment_length={ppo_overrides['rollout_fragment_length']}, "
            f"sgd_minibatch_size={ppo_overrides['sgd_minibatch_size']}"
        )
        config.update(ppo_overrides)
    return config


def evaluation_json_path(local_dir: str, run_name: str) -> str:
    return os.path.join(local_dir, "evaluations", f"{run_name}_vs_ceia_baseline.json")


def coerce_result_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped == "True":
            return True
        if stripped == "False":
            return False
        try:
            if any(char in stripped for char in (".", "e", "E")):
                return float(stripped)
            return int(stripped)
        except ValueError:
            return value
    return value


def load_latest_result(trial_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not trial_dir:
        return None

    result_json_path = os.path.join(trial_dir, "result.json")
    if os.path.isfile(result_json_path):
        last_line = None
        with open(result_json_path) as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is not None:
            return json.loads(last_line)

    progress_csv_path = os.path.join(trial_dir, "progress.csv")
    if not os.path.isfile(progress_csv_path):
        return None

    last_row = None
    with open(progress_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if last_row is None:
        return None
    return {key: coerce_result_value(value) for key, value in last_row.items()}


def latest_checkpoint_path_or_none(trial_dir: Optional[str]) -> Optional[str]:
    if not trial_dir:
        return None
    try:
        return latest_checkpoint_path(trial_dir)
    except FileNotFoundError:
        return None


def load_cached_eval_payload(
    local_dir: str,
    run_name: str,
    checkpoint_path: Optional[str] = None,
    eval_episodes: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    payload_path = evaluation_json_path(local_dir, run_name)
    if not os.path.isfile(payload_path):
        return None

    with open(payload_path) as f:
        payload = json.load(f)

    if checkpoint_path is not None and payload.get("checkpoint_path") != checkpoint_path:
        return None
    if eval_episodes is not None and payload.get("eval_episodes") != eval_episodes:
        return None
    return payload


def trial_result_run_name(trial) -> str:
    return trial.config["env_config"]["_sweep_meta"]["run_name"]


def infer_local_trial_status(
    trial_dir: Optional[str],
    last_result: Optional[Mapping[str, Any]],
) -> str:
    if not trial_dir or not os.path.isdir(trial_dir):
        return "PENDING"

    error_path = os.path.join(trial_dir, "error.txt")
    if os.path.isfile(error_path):
        return "ERROR"

    done = coerce_result_value((last_result or {}).get("done"))
    if done is True:
        return "TERMINATED"

    if last_result is not None or os.path.isfile(os.path.join(trial_dir, "progress.csv")):
        return "RUNNING"
    return "PENDING"


def find_trial_dir_for_run_name(experiment_dir: str, run_name: str) -> Optional[str]:
    exact_path = os.path.join(experiment_dir, run_name)
    if os.path.isdir(exact_path):
        return exact_path

    candidates = [
        path
        for path in glob(os.path.join(experiment_dir, f"{run_name}*"))
        if os.path.isdir(path)
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (
            latest_recorded_timesteps(path) or -1,
            os.path.getmtime(path),
        ),
    )


def extract_eval_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = payload.get("result", {})
    policies = result.get("policies", {})
    agent_key = payload.get("agent_module", CHECKPOINT_AGENT_MODULE)
    baseline_key = payload.get("baseline_module", CEIA_AGENT_MODULE)

    agent_stats = policies.get(agent_key, {})
    baseline_stats = policies.get(baseline_key, {})
    return {
        "eval_agent_module": agent_key,
        "eval_baseline_module": baseline_key,
        "eval_episodes": payload.get("eval_episodes"),
        "vs_ceia_win_rate": agent_stats.get("policy_win_rate"),
        "vs_ceia_wins": agent_stats.get("policy_wins"),
        "vs_ceia_losses": agent_stats.get("policy_losses"),
        "vs_ceia_draws": agent_stats.get("policy_draws"),
        "vs_ceia_reward_mean": agent_stats.get("policy_reward_mean"),
        "ceia_win_rate": baseline_stats.get("policy_win_rate"),
        "eval_episode_reward_mean": result.get("episode_reward_mean"),
        "eval_episode_reward_max": result.get("episode_reward_max"),
        "eval_episode_reward_min": result.get("episode_reward_min"),
        "eval_episode_len_mean": result.get("episode_len_mean"),
    }


def build_summary_row_from_metadata(
    trial_spec: Mapping[str, Any],
    local_dir: str,
    status: str,
    trial_dir: Optional[str] = None,
    last_result: Optional[Mapping[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    eval_payload: Optional[Mapping[str, Any]] = None,
    error_file: Optional[str] = None,
) -> Dict[str, Any]:
    progress_csv = os.path.join(trial_dir, "progress.csv") if trial_dir else None
    cached_eval_payload = eval_payload
    if cached_eval_payload is None:
        cached_eval_payload = load_cached_eval_payload(
            local_dir=local_dir,
            run_name=trial_spec["run_name"],
            checkpoint_path=checkpoint_path,
        )

    row = {
        "run_name": trial_spec["run_name"],
        "label": trial_spec["label"],
        "status": status,
        "trial_dir": trial_dir,
        "progress_csv": progress_csv
        if progress_csv is not None and os.path.isfile(progress_csv)
        else None,
        "checkpoint_path": checkpoint_path,
        "error_file": error_file,
        "train_base_port": trial_spec["train_base_port"],
        "eval_base_port": trial_spec["eval_base_port"],
        "timesteps_total": coerce_result_value((last_result or {}).get("timesteps_total"))
        or latest_recorded_timesteps(trial_dir or ""),
        "training_iteration": coerce_result_value(
            (last_result or {}).get("training_iteration")
        ),
        "train_episode_reward_mean": coerce_result_value(
            (last_result or {}).get("episode_reward_mean")
        ),
        "train_episode_len_mean": coerce_result_value(
            (last_result or {}).get("episode_len_mean")
        ),
        "eval_json_path": evaluation_json_path(local_dir, trial_spec["run_name"])
        if cached_eval_payload is not None
        else None,
    }
    row.update(flatten_dict(trial_spec["reward_shaping"], prefix="reward_shaping"))
    if cached_eval_payload is not None:
        row.update(extract_eval_metrics(cached_eval_payload))
    return row


def build_summary_row(
    trial,
    trial_spec: Mapping[str, Any],
    checkpoint_path: Optional[str],
    eval_payload: Optional[Mapping[str, Any]],
    local_dir: str,
) -> Dict[str, Any]:
    return build_summary_row_from_metadata(
        trial_spec=trial_spec,
        local_dir=local_dir,
        status=trial.status,
        trial_dir=trial.logdir,
        last_result=trial.last_result or {},
        checkpoint_path=checkpoint_path,
        eval_payload=eval_payload,
        error_file=getattr(trial, "error_file", None),
    )


def write_summary_csv(rows: List[Dict[str, Any]], csv_path: str):
    preferred_fields = [
        "run_name",
        "label",
        "status",
        "timesteps_total",
        "training_iteration",
        "train_episode_reward_mean",
        "train_episode_len_mean",
        "vs_ceia_win_rate",
        "vs_ceia_reward_mean",
        "vs_ceia_wins",
        "vs_ceia_losses",
        "vs_ceia_draws",
        "eval_episodes",
        "train_base_port",
        "eval_base_port",
        "trial_dir",
        "progress_csv",
        "checkpoint_path",
        "eval_json_path",
        "error_file",
    ]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in preferred_fields
        }
    )
    fieldnames = preferred_fields + extra_fields

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sweep_plan(
    trial_specs: List[Dict[str, Any]],
    plan_path: str,
    sweep_spec_path: str,
    probe_base_port: int,
):
    payload = {
        "sweep_spec_path": sweep_spec_path,
        "probe_base_port": probe_base_port,
        "trial_count": len(trial_specs),
        "trials": [
            {
                "index": trial_spec["index"],
                "label": trial_spec["label"],
                "run_name": trial_spec["run_name"],
                "source": trial_spec["source"],
                "train_base_port": trial_spec["train_base_port"],
                "eval_base_port": trial_spec["eval_base_port"],
                "reward_shaping": trial_spec["reward_shaping"],
            }
            for trial_spec in trial_specs
        ],
    }
    with open(plan_path, "w") as f:
        json.dump(normalize_for_json(payload), f, indent=2)


class LiveSweepSummaryCallback(Callback):
    def __init__(
        self,
        trial_specs: List[Dict[str, Any]],
        local_dir: str,
        eval_episodes: int,
        skip_eval: bool,
        enable_live_eval: bool,
        summary_csv_path: str,
    ):
        self.local_dir = local_dir
        self.eval_episodes = eval_episodes
        self.skip_eval = skip_eval
        self.enable_live_eval = enable_live_eval
        self.summary_csv_path = summary_csv_path
        self.run_names = [trial_spec["run_name"] for trial_spec in trial_specs]
        self.trial_specs_by_name = {
            trial_spec["run_name"]: trial_spec for trial_spec in trial_specs
        }
        self.rows_by_name = {
            trial_spec["run_name"]: build_summary_row_from_metadata(
                trial_spec=trial_spec,
                local_dir=local_dir,
                status="PENDING",
            )
            for trial_spec in trial_specs
        }
        self._write()

    def _ordered_rows(self) -> List[Dict[str, Any]]:
        return [self.rows_by_name[run_name] for run_name in self.run_names]

    def _write(self):
        write_summary_csv(self._ordered_rows(), self.summary_csv_path)

    def _update_row(self, trial, status: Optional[str] = None):
        run_name = trial_result_run_name(trial)
        trial_spec = self.trial_specs_by_name[run_name]
        checkpoint_path = latest_checkpoint_path_or_none(trial.logdir)
        row = build_summary_row_from_metadata(
            trial_spec=trial_spec,
            local_dir=self.local_dir,
            status=status or trial.status,
            trial_dir=trial.logdir,
            last_result=trial.last_result or {},
            checkpoint_path=checkpoint_path,
            error_file=getattr(trial, "error_file", None),
        )
        self.rows_by_name[run_name] = row
        self._write()

    def on_trial_start(self, iteration: int, trials: List[Any], trial, **info):
        self._update_row(trial)

    def on_trial_save(self, iteration: int, trials: List[Any], trial, **info):
        self._update_row(trial)

    def on_trial_complete(self, iteration: int, trials: List[Any], trial, **info):
        run_name = trial_result_run_name(trial)
        trial_spec = self.trial_specs_by_name[run_name]
        checkpoint_path = latest_checkpoint_path_or_none(trial.logdir)
        eval_payload = None

        if self.enable_live_eval and (not self.skip_eval) and checkpoint_path is not None:
            print(
                f"\n=== Live CEIA evaluation for {run_name} "
                f"(base_port={trial_spec['eval_base_port']}) ==="
            )
            eval_payload = evaluate_against_ceia(
                checkpoint_path=checkpoint_path,
                run_name=run_name,
                local_dir=self.local_dir,
                eval_episodes=self.eval_episodes,
                eval_base_port=trial_spec["eval_base_port"],
            )

        self.rows_by_name[run_name] = build_summary_row_from_metadata(
            trial_spec=trial_spec,
            local_dir=self.local_dir,
            status="TERMINATED",
            trial_dir=trial.logdir,
            last_result=trial.last_result or {},
            checkpoint_path=checkpoint_path,
            eval_payload=eval_payload,
            error_file=getattr(trial, "error_file", None),
        )
        self._write()

    def on_trial_error(self, iteration: int, trials: List[Any], trial, **info):
        self._update_row(trial, status="ERROR")


class FractionalGPUPPO(PPOTrainer):
    """PPO variant that supports fractional num_gpus for Ray scheduling.

    default_resource_request() is a class method called before __init__, so
    it correctly reads the fractional value from the config for placement-group
    scheduling. setup() then converts num_gpus to an integer before RLlib
    creates policies, avoiding the range(float) TypeError in torch_policy.py.
    """

    def setup(self, config):
        config["num_gpus"] = int(config["num_gpus"] > 0)
        super().setup(config)


def init_ray(args):
    desired_num_cpus = max(1, args.parallel_trials) * max(1, args.num_workers + 1)
    detected_num_cpus = os.cpu_count() or desired_num_cpus
    ray_num_cpus = (
        args.ray_num_cpus
        if args.ray_num_cpus is not None
        else min(detected_num_cpus, desired_num_cpus)
    )
    ray_temp_dir = os.path.join(
        tempfile.gettempdir(),
        f"ray_reward_sweep_{os.getpid()}",
    )

    ray_kwargs = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "_node_ip_address": "127.0.0.1",
        "_temp_dir": ray_temp_dir,
        "num_cpus": ray_num_cpus,
    }
    if args.ray_num_gpus is not None:
        ray_kwargs["num_gpus"] = args.ray_num_gpus

    print(
        f"[ray] starting shared cluster with num_cpus={ray_num_cpus}, "
        f"temp_dir={ray_temp_dir}"
        + (
            f", num_gpus={ray_kwargs['num_gpus']}"
            if "num_gpus" in ray_kwargs
            else ""
        )
    )
    ray.init(**ray_kwargs)


def run_ceia_eval_task(
    checkpoint_path: str,
    run_name: str,
    local_dir: str,
    eval_episodes: int,
    eval_base_port: int,
) -> Dict[str, Any]:
    return evaluate_against_ceia(
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        local_dir=local_dir,
        eval_episodes=eval_episodes,
        eval_base_port=eval_base_port,
    )


def evaluate_trials(
    trials: List[Any],
    trial_specs: List[Dict[str, Any]],
    local_dir: str,
    eval_episodes: int,
    skip_eval: bool,
    eval_parallelism: int,
    summary_csv_path: str,
):
    if eval_parallelism < 1:
        raise ValueError("--eval-parallelism must be at least 1.")

    rows_by_name: Dict[str, Dict[str, Any]] = {}
    ordered_run_names = [trial_spec["run_name"] for trial_spec in trial_specs]
    trials_by_name = {trial_result_run_name(trial): trial for trial in trials}

    def write_partial_rows():
        ordered_rows = [
            rows_by_name[run_name]
            for run_name in ordered_run_names
            if run_name in rows_by_name
        ]
        write_summary_csv(ordered_rows, summary_csv_path)

    ray.shutdown()
    eval_tasks: List[Dict[str, Any]] = []
    for trial_spec in trial_specs:
        trial = trials_by_name[trial_spec["run_name"]]
        checkpoint_path = None

        try:
            checkpoint_path = latest_checkpoint_path(trial.logdir)
        except FileNotFoundError:
            checkpoint_path = None

        should_eval = (
            not skip_eval
            and trial.status == "TERMINATED"
            and checkpoint_path is not None
        )
        if should_eval:
            eval_tasks.append(
                {
                    "trial": trial,
                    "trial_spec": trial_spec,
                    "checkpoint_path": checkpoint_path,
                }
            )
            continue

        row = build_summary_row(
            trial=trial,
            trial_spec=trial_spec,
            checkpoint_path=checkpoint_path,
            eval_payload=None,
            local_dir=local_dir,
        )
        rows_by_name[trial_spec["run_name"]] = row
        write_partial_rows()

    if eval_tasks:
        if eval_parallelism == 1:
            for task in eval_tasks:
                trial = task["trial"]
                trial_spec = task["trial_spec"]
                checkpoint_path = task["checkpoint_path"]

                print(
                    f"\n=== Evaluating {trial_spec['run_name']} vs ceia_baseline_agent "
                    f"(base_port={trial_spec['eval_base_port']}) ==="
                )
                try:
                    eval_payload = run_ceia_eval_task(
                        checkpoint_path=checkpoint_path,
                        run_name=trial_spec["run_name"],
                        local_dir=local_dir,
                        eval_episodes=eval_episodes,
                        eval_base_port=trial_spec["eval_base_port"],
                    )
                finally:
                    ray.shutdown()

                row = build_summary_row(
                    trial=trial,
                    trial_spec=trial_spec,
                    checkpoint_path=checkpoint_path,
                    eval_payload=eval_payload,
                    local_dir=local_dir,
                )
                rows_by_name[trial_spec["run_name"]] = row
                write_partial_rows()
        else:
            print(
                f"[eval] Running {len(eval_tasks)} CEIA evaluations in parallel "
                f"(max_workers={eval_parallelism})."
            )
            max_workers = min(eval_parallelism, len(eval_tasks))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for task in eval_tasks:
                    trial_spec = task["trial_spec"]
                    checkpoint_path = task["checkpoint_path"]
                    future = executor.submit(
                        run_ceia_eval_task,
                        checkpoint_path,
                        trial_spec["run_name"],
                        local_dir,
                        eval_episodes,
                        trial_spec["eval_base_port"],
                    )
                    future_to_task[future] = task

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    trial = task["trial"]
                    trial_spec = task["trial_spec"]
                    checkpoint_path = task["checkpoint_path"]

                    try:
                        eval_payload = future.result()
                    except Exception as exc:
                        for pending in future_to_task:
                            pending.cancel()
                        raise RuntimeError(
                            f"Parallel CEIA evaluation failed for {trial_spec['run_name']}."
                        ) from exc

                    row = build_summary_row(
                        trial=trial,
                        trial_spec=trial_spec,
                        checkpoint_path=checkpoint_path,
                        eval_payload=eval_payload,
                        local_dir=local_dir,
                    )
                    rows_by_name[trial_spec["run_name"]] = row
                    write_partial_rows()

    rows = [rows_by_name[trial_spec["run_name"]] for trial_spec in trial_specs]
    return rows


def main():
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    configure_visible_gpus(args)
    validate_resource_args(args)
    if args.eval_parallelism < 1:
        raise ValueError("--eval-parallelism must be at least 1.")

    min_reasonable_stride = max(16, args.num_workers * NUM_ENVS_PER_WORKER + 8)
    if args.port_stride < min_reasonable_stride:
        raise ValueError(
            f"--port-stride={args.port_stride} is too small for "
            f"{args.num_workers} workers and {NUM_ENVS_PER_WORKER} envs/worker. "
            f"Use at least {min_reasonable_stride}."
        )

    sweep_spec = load_sweep_spec(args.sweep_spec)
    trial_specs = build_trial_specs(sweep_spec)
    if args.trial_limit is not None:
        trial_specs = trial_specs[: args.trial_limit]

    probe_base_port = assign_ports(
        trial_specs=trial_specs,
        base_port_start=args.base_port_start,
        port_stride=args.port_stride,
        eval_base_port_start=args.eval_base_port_start,
        probe_base_port=args.probe_base_port,
    )

    plan_path = os.path.join(args.local_dir, "reward_sweep_plan.json")
    summary_csv_path = os.path.join(args.local_dir, "reward_sweep_summary.csv")
    write_sweep_plan(
        trial_specs=trial_specs,
        plan_path=plan_path,
        sweep_spec_path=os.path.abspath(args.sweep_spec),
        probe_base_port=probe_base_port,
    )

    print(
        f"[plan] {len(trial_specs)} trials written to {plan_path}\n"
        f"[ports] training starts at {args.base_port_start}, "
        f"eval starts at "
        f"{trial_specs[0]['eval_base_port'] if trial_specs else 'n/a'}, "
        f"probe base port is {probe_base_port}"
    )
    if args.dry_run:
        print("[dry-run] Exiting before training.")
        return

    tune.registry.register_env(ENV_NAME, create_rllib_env_shaped)

    init_ray(args)
    try:
        temp_env = open_probe_env(create_rllib_env_shaped, probe_base_port)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        temp_env.close()

        config = build_tune_config(
            obs_space=obs_space,
            act_space=act_space,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            trial_specs=trial_specs,
            timesteps=args.timesteps,
        )
        enable_live_eval = args.eval_parallelism == 1
        if not args.skip_eval and not enable_live_eval:
            print(
                f"[eval] Disabling live per-trial eval and deferring to "
                f"post-training parallel eval (eval_parallelism={args.eval_parallelism})."
            )
        live_summary_callback = LiveSweepSummaryCallback(
            trial_specs=trial_specs,
            local_dir=args.local_dir,
            eval_episodes=args.eval_episodes,
            skip_eval=args.skip_eval,
            enable_live_eval=enable_live_eval,
            summary_csv_path=summary_csv_path,
        )

        print(
            f"\n=== Starting reward sweep: {len(trial_specs)} trials, "
            f"timesteps={args.timesteps}, parallel_trials~{args.parallel_trials} ==="
        )
        analysis = tune.run(
            FractionalGPUPPO,
            name=args.experiment_name,
            config=config,
            stop={"timesteps_total": args.timesteps},
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=args.local_dir,
            queue_trials=True,
            raise_on_failed_trial=False,
            trial_name_creator=trial_run_name,
            trial_dirname_creator=build_trial_dirname,
            callbacks=[live_summary_callback],
        )
        trials = analysis.trials
    finally:
        ray.shutdown()

    rows = evaluate_trials(
        trials=trials,
        trial_specs=trial_specs,
        local_dir=args.local_dir,
        eval_episodes=args.eval_episodes,
        skip_eval=args.skip_eval,
        eval_parallelism=args.eval_parallelism,
        summary_csv_path=summary_csv_path,
    )

    terminated = sum(1 for row in rows if row["status"] == "TERMINATED")
    errored = sum(1 for row in rows if row["status"] != "TERMINATED")
    print(
        "\nDone.\n"
        f"[summary] {summary_csv_path}\n"
        f"[plan]    {plan_path}\n"
        f"[trials]  terminated={terminated}, non_terminated={errored}"
    )


if __name__ == "__main__":
    main()
