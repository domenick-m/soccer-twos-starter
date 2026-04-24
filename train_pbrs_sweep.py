"""
Sweep over GoalAwarePBRS hyperparameters for SoccerTwos PPO.

Modelled after the working train_reward_sweep.py infrastructure but uses
the classmate's potential-based reward shaping wrapper instead of the
custom RewardShapingWrapper.

Usage (smoke test):
    python train_pbrs_sweep.py --timesteps 100 --num-workers 2 --gpu-ids 0,1 --skip-eval

Full run:
    python train_pbrs_sweep.py --timesteps 5000000 --num-workers 4 \
        --parallel-trials 4 --gpu-ids 0,1 --num-gpus 0.5
"""

import argparse
import copy
import json
import math
import os
import socket
import tempfile
from typing import Any, Dict, List, Mapping, Optional

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from goal_aware_pbrs import create_rllib_env_pbrs


NUM_ENVS_PER_WORKER = 3
ENV_NAME = "SoccerPBRS"
DEFAULT_LOCAL_DIR = os.path.abspath("./ray_results_pbrs_sweep")
DEFAULT_EXPERIMENT_NAME = "PPO_pbrs_sweep"


# ── PBRS sweep configs ──────────────────────────────────────────────────────
# Each config is a dict of GoalAwarePBRSWrapper kwargs.
# We sweep around the classmate's defaults to find robust settings.

SWEEP_CONFIGS = [
    # ── Baseline & Defaults ──
    {"label": "sparse_only", "pbrs": {"alpha": 0.0, "beta": 0.0, "potential_scale": 0.0, "kick_base": 0.0, "kick_goal_bonus": 0.0}},
    {"label": "classmate_default", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    
    # ── Sweep Beta (Goal pursuit aggressiveness) ──
    {"label": "beta_0.0_ball_only", "pbrs": {"alpha": 1.0, "beta": 0.0, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "beta_0.1_weak_goal", "pbrs": {"alpha": 1.0, "beta": 0.1, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "beta_0.6_strong_goal", "pbrs": {"alpha": 1.0, "beta": 0.6, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "beta_0.9_extreme_goal", "pbrs": {"alpha": 1.0, "beta": 0.9, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.06}},

    # ── Sweep Potential Scale (Overall magnitude of PBRS shaping) ──
    {"label": "scale_0.001_tiny", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.001, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "scale_0.005_small", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.005, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "scale_0.03_large", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.03, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
    {"label": "scale_0.1_huge", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.1, "kick_base": 0.04, "kick_goal_bonus": 0.06}},

    # ── Sweep Kick Base (Flat reward for touching the ball) ──
    {"label": "kick_base_0.0_none", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.0, "kick_goal_bonus": 0.06}},
    {"label": "kick_base_0.02_low", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.02, "kick_goal_bonus": 0.06}},
    {"label": "kick_base_0.08_high", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.08, "kick_goal_bonus": 0.06}},
    {"label": "kick_base_0.15_extreme", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.15, "kick_goal_bonus": 0.06}},

    # ── Sweep Kick Goal Bonus (Directional reward for kicking toward goal) ──
    {"label": "kick_bonus_0.0_none", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.0}},
    {"label": "kick_bonus_0.03_low", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.03}},
    {"label": "kick_bonus_0.12_high", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.12}},
    {"label": "kick_bonus_0.20_extreme", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.04, "kick_goal_bonus": 0.20}},

    # ── Pure Components ──
    {"label": "pure_pbrs_no_kick", "pbrs": {"alpha": 1.0, "beta": 0.3, "potential_scale": 0.01, "kick_base": 0.0, "kick_goal_bonus": 0.0}},
    {"label": "pure_kick_no_pbrs", "pbrs": {"alpha": 0.0, "beta": 0.0, "potential_scale": 0.0, "kick_base": 0.04, "kick_goal_bonus": 0.06}},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative value.")
    return parsed


def rllib_policy_num_gpus(num_gpus: float) -> int:
    return int(num_gpus > 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=parse_nonnegative_float, default=0.5,
                        help="GPUs per trial (fractional ok).")
    parser.add_argument("--parallel-trials", type=int, default=2)
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=float, default=None)
    parser.add_argument("--gpu-ids", default=None,
                        help="Comma-separated GPU ids, e.g. '0,1'.")
    parser.add_argument("--base-port-start", type=int, default=55000)
    parser.add_argument("--port-stride", type=int, default=128)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def parse_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg is None:
        return None
    parsed = [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]
    if not parsed:
        raise ValueError("No valid GPU ids in --gpu-ids.")
    return parsed


def configure_visible_gpus(args):
    selected = parse_gpu_ids(args.gpu_ids)
    if selected is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in selected)
    if args.ray_num_gpus is None:
        args.ray_num_gpus = float(len(selected))
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


def port_is_free(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_free_port(start, stride=128):
    candidate = start
    while candidate + stride - 1 <= 65535:
        if all(port_is_free(candidate + i) for i in range(stride)):
            return candidate
        candidate += stride
    raise RuntimeError(f"No free port block of size {stride} from {start}")


class FractionalGPUPPO(PPOTrainer):
    """PPO that supports fractional num_gpus for Ray scheduling."""
    _allow_unknown_configs = True

    @classmethod
    def default_resource_request(cls, config):
        rc = dict(config)
        meta = rc.get("env_config", {}).get("_sweep_meta", {}) if isinstance(rc.get("env_config"), dict) else {}
        frac = meta.get("resource_num_gpus")
        if frac is not None:
            rc["num_gpus"] = frac
        return PPOTrainer.default_resource_request(rc)

    def setup(self, config):
        assigned = config.get("env_config", {}).get("_sweep_meta", {}).get("assigned_cuda_device")
        if assigned is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned)
        config["num_gpus"] = rllib_policy_num_gpus(config["num_gpus"])
        super().setup(config)

    def _init(self, config, env_creator):
        config["num_gpus"] = rllib_policy_num_gpus(config["num_gpus"])
        self.config["num_gpus"] = rllib_policy_num_gpus(self.config["num_gpus"])
        super()._init(config, env_creator)


def small_timestep_overrides(timesteps, num_workers):
    if timesteps >= 4000:
        return {}
    eff = max(1, num_workers) * max(1, NUM_ENVS_PER_WORKER)
    tbs = max(eff, max(1, timesteps))
    rfl = max(1, int(math.ceil(tbs / float(eff))))
    sgd = min(128, tbs)
    return {"train_batch_size": tbs, "rollout_fragment_length": rfl, "sgd_minibatch_size": sgd}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    configure_visible_gpus(args)

    configs = SWEEP_CONFIGS
    if args.trial_limit is not None:
        configs = configs[:args.trial_limit]

    # Assign ports
    next_port = args.base_port_start
    for i, cfg in enumerate(configs):
        cfg["index"] = i
        cfg["run_name"] = f"{i:03d}_{cfg['label']}"
        cfg["train_base_port"] = find_free_port(next_port, args.port_stride)
        next_port = cfg["train_base_port"] + args.port_stride

    # Assign GPUs round-robin
    selected_gpus = parse_gpu_ids(args.gpu_ids)
    if selected_gpus:
        for cfg in configs:
            cfg["assigned_cuda_device"] = str(selected_gpus[cfg["index"] % len(selected_gpus)])

    # Probe port
    probe_port = find_free_port(next_port + args.port_stride * 2, args.port_stride)

    print(f"\n=== PBRS Sweep: {len(configs)} configs, {args.timesteps} timesteps ===")
    for cfg in configs:
        print(f"  [{cfg['index']}] {cfg['label']}: {cfg['pbrs']}")

    # Build env_configs for grid_search
    env_configs = []
    for cfg in configs:
        ec = {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.multiagent_player,
            "base_port": cfg["train_base_port"],
            "_sweep_meta": {
                "index": cfg["index"],
                "label": cfg["label"],
                "run_name": cfg["run_name"],
                "assigned_cuda_device": cfg.get("assigned_cuda_device"),
                "resource_num_gpus": args.num_gpus,
            },
        }
        # Merge PBRS kwargs into env_config (consumed by create_rllib_env_pbrs)
        ec.update(cfg["pbrs"])
        env_configs.append(ec)

    # Init Ray
    desired_cpus = max(1, args.parallel_trials) * max(1, args.num_workers + 1)
    detected_cpus = os.cpu_count() or desired_cpus
    ray_cpus = args.ray_num_cpus if args.ray_num_cpus is not None else min(detected_cpus, desired_cpus)
    ray_temp = os.path.join(tempfile.gettempdir(), f"ray_pbrs_sweep_{os.getpid()}")

    ray_kwargs = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "_node_ip_address": "127.0.0.1",
        "_temp_dir": ray_temp,
        "num_cpus": ray_cpus,
    }
    if args.ray_num_gpus is not None:
        ray_kwargs["num_gpus"] = args.ray_num_gpus

    print(f"[ray] num_cpus={ray_cpus}, temp={ray_temp}")
    ray.init(**ray_kwargs)

    tune.registry.register_env(ENV_NAME, create_rllib_env_pbrs)

    # Probe env for obs/action spaces
    from mlagents_envs.exception import UnityWorkerInUseException
    temp_env = None
    for offset in range(32):
        try:
            temp_env = create_rllib_env_pbrs({
                "variation": EnvType.multiagent_player,
                "base_port": probe_port,
                "worker_id": 900 + offset,
            })
            break
        except UnityWorkerInUseException:
            continue
    if temp_env is None:
        raise RuntimeError("Could not open probe env")

    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # Build Tune config
    tune_config = {
        "num_gpus": rllib_policy_num_gpus(args.num_gpus),
        "num_workers": args.num_workers,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",
        "multiagent": {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda _: "default",
            "policies_to_train": ["default"],
        },
        "env": ENV_NAME,
        "env_config": tune.grid_search(env_configs),
    }
    overrides = small_timestep_overrides(args.timesteps, args.num_workers)
    if overrides:
        print(f"[ppo] small-timestep overrides: {overrides}")
        tune_config.update(overrides)

    def trial_name(trial):
        return trial.config["env_config"]["_sweep_meta"]["run_name"]

    def trial_dirname(trial):
        return trial.config["env_config"]["_sweep_meta"]["run_name"]

    try:
        print(f"\n=== Launching {len(configs)} trials ===\n")
        analysis = tune.run(
            FractionalGPUPPO,
            name=args.experiment_name,
            config=tune_config,
            stop={"timesteps_total": args.timesteps},
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=args.local_dir,
            queue_trials=True,
            raise_on_failed_trial=False,
            trial_name_creator=trial_name,
            trial_dirname_creator=trial_dirname,
        )

        print("\n=== Results ===")
        for trial in analysis.trials:
            meta = trial.config["env_config"]["_sweep_meta"]
            result = trial.last_result or {}
            print(f"  {meta['run_name']}: status={trial.status}, "
                  f"timesteps={result.get('timesteps_total', '?')}, "
                  f"reward_mean={result.get('episode_reward_mean', '?')}")
    finally:
        ray.shutdown()

    print(f"\nDone. Results in: {args.local_dir}")


if __name__ == "__main__":
    main()
