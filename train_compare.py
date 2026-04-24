"""
Trains PPO on SoccerTwos twice under identical hyperparameters:
  (a) baseline/control - default environment reward
  (b) shaped           - configurable dense reward shaping
                         (see reward_shaping.DEFAULT_REWARD_SHAPING_CONFIG)

Important: `baseline` here means the unmodified training condition. Training
does not depend on the packaged `ceia_baseline_agent` used by
`python -m soccer_twos.watch`. The packaged baseline is only used when
`--evaluate-vs-ceia` is enabled.

Optionally, after training, it can evaluate the latest checkpoint from each run
against the packaged `ceia_baseline_agent` and write a JSON summary with win
rates and rewards under `./ray_results_compare/evaluations/`.

By default, the baseline/control run is reused if a prior run already exists
under `./ray_results_compare/PPO_baseline/` with a checkpoint and at least the
requested number of timesteps. Use `--retrain-baseline` to force a fresh run.

Writes each run's progress.csv to `./ray_results_compare/PPO_{baseline,shaped}/...`
so that plot_compare.py can overlay the learning curves.

Usage
-----
    python train_compare.py --variant both --timesteps 500000 --num-workers 4
    python train_compare.py --variant baseline
    python train_compare.py --variant shaped
"""

import argparse
import collections
import csv
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional

from mlagents_envs.exception import UnityWorkerInUseException
import ray
from ray import tune
from ray.tune.logger import pretty_print
from soccer_twos import EnvType

from utils import create_rllib_env
from reward_shaping import create_rllib_env_shaped


NUM_ENVS_PER_WORKER = 3
CEIA_AGENT_MODULE = "ceia_baseline_agent"
CHECKPOINT_AGENT_MODULE = "compare_checkpoint_agent"
PROBE_WORKER_ID_START = 1000
PROBE_WORKER_ID_ATTEMPTS = 32
EVAL_AGENT1_PORT_OFFSET = 0
EVAL_AGENT2_PORT_OFFSET = 32
EVAL_MATCH_PORT_OFFSET = 64


def offset_base_port(base_port: Optional[int], offset: int) -> Optional[int]:
    if base_port is None:
        return None
    return base_port + offset


def build_config(
    env_name: str,
    obs_space,
    act_space,
    num_workers: int,
    num_gpus: int,
    base_port: Optional[int],
):
    env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player,
    }
    if base_port is not None:
        env_config["base_port"] = base_port

    return {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",
        "multiagent": {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": tune.function(lambda _: "default"),
            "policies_to_train": ["default"],
        },
        "env": env_name,
        "env_config": env_config,
    }


def open_probe_env(env_factory, base_port: Optional[int]):
    probe_config = {"variation": EnvType.multiagent_player}
    if base_port is not None:
        probe_config["base_port"] = base_port

    last_error = None
    for offset in range(PROBE_WORKER_ID_ATTEMPTS):
        try:
            return env_factory(
                {
                    **probe_config,
                    "worker_id": PROBE_WORKER_ID_START + offset,
                }
            )
        except UnityWorkerInUseException as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to open temporary probe environment.")


def _checkpoint_sort_key(path: str):
    dirname = os.path.basename(os.path.dirname(path))
    suffix = dirname.split("_")[-1]
    try:
        return int(suffix)
    except ValueError:
        return -1


def latest_checkpoint_path(trial_dir: str) -> str:
    pattern = os.path.join(trial_dir, "checkpoint_*", "checkpoint-*")
    matches = sorted(
        [
            path
            for path in glob(pattern)
            if os.path.isfile(path) and not path.endswith(".tune_metadata")
        ],
        key=_checkpoint_sort_key,
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoints found under {trial_dir}")
    return matches[-1]


def latest_recorded_timesteps(trial_dir: str) -> Optional[int]:
    progress_path = os.path.join(trial_dir, "progress.csv")
    if not os.path.isfile(progress_path):
        return None

    latest_value = None
    with open(progress_path, newline="") as f:
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


def find_reusable_run(run_name: str, local_dir: str, min_timesteps: int):
    run_dir = os.path.join(local_dir, run_name)
    if not os.path.isdir(run_dir):
        return None

    candidates = []
    for trial_dir in glob(os.path.join(run_dir, "*")):
        if not os.path.isdir(trial_dir):
            continue

        try:
            checkpoint_path = latest_checkpoint_path(trial_dir)
        except FileNotFoundError:
            continue

        timesteps_total = latest_recorded_timesteps(trial_dir)
        if timesteps_total is None or timesteps_total < min_timesteps:
            continue

        candidates.append(
            {
                "run_name": run_name,
                "trial_dir": trial_dir,
                "checkpoint_path": checkpoint_path,
                "timesteps_total": timesteps_total,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: (
            candidate["timesteps_total"],
            os.path.getmtime(candidate["checkpoint_path"]),
        ),
    )


def normalize_for_json(value: Any):
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _reward_summary(rewards: List[float], prefix: str) -> Dict[str, Any]:
    results = []
    for reward in rewards:
        if reward > 0:
            results.append(1)
        elif reward < 0:
            results.append(-1)
        else:
            results.append(0)

    total_games = len(rewards)
    return {
        f"{prefix}_reward_min": min(rewards) if rewards else None,
        f"{prefix}_reward_max": max(rewards) if rewards else None,
        f"{prefix}_reward_mean": (sum(rewards) / total_games) if rewards else None,
        f"{prefix}_total_games": total_games,
        f"{prefix}_wins": results.count(1),
        f"{prefix}_losses": results.count(-1),
        f"{prefix}_draws": results.count(0),
        f"{prefix}_win_rate": (results.count(1) / total_games) if rewards else None,
    }


def summarize_episodes_safe(
    episodes: List[Dict[str, Any]],
    agent_1_name: str,
    agent_2_name: str,
) -> Dict[str, Any]:
    episode_lengths: List[int] = []
    episode_rewards: List[float] = []
    hist_stats = {
        agent_name: {
            "rewards": [],
            "blue_team": collections.defaultdict(list),
            "orange_team": collections.defaultdict(list),
        }
        for agent_name in (agent_1_name, agent_2_name)
    }

    for episode in episodes:
        episode_lengths.append(episode["episode_length"])
        episode_rewards.append(episode["agent_1_reward"] + episode["agent_2_reward"])

        for agent_id, agent_name in [(1, agent_1_name), (2, agent_2_name)]:
            team = episode[f"team_agent_{agent_id}"]
            reward = episode[f"agent_{agent_id}_reward"]
            hist_stats[agent_name]["rewards"].append(reward)
            hist_stats[agent_name][team]["rewards"].append(reward)

    hist_stats["episode_reward"] = episode_rewards
    hist_stats["episode_lengths"] = episode_lengths

    policies = {}
    for agent_name in (agent_1_name, agent_2_name):
        agent_rewards = hist_stats[agent_name]["rewards"]
        policies[agent_name] = _reward_summary(agent_rewards, "policy")
        for team in ("blue_team", "orange_team"):
            team_rewards = hist_stats[agent_name][team]["rewards"]
            policies[agent_name][team] = _reward_summary(team_rewards, f"policy_{team}")

    return {
        "episode_reward_max": max(episode_rewards) if episode_rewards else None,
        "episode_reward_min": min(episode_rewards) if episode_rewards else None,
        "episode_reward_mean": (
            sum(episode_rewards) / len(episode_rewards) if episode_rewards else None
        ),
        "episode_len_mean": (
            sum(episode_lengths) / len(episode_lengths) if episode_lengths else None
        ),
        "episodes_this_eval": len(episodes),
        "policies": policies,
        "hist_stats": dict(hist_stats),
    }


def evaluate_matchup(
    agent1_module_name: str,
    agent2_module_name: str,
    n_episodes: int,
    base_port: Optional[int],
) -> Dict[str, Any]:
    import soccer_twos
    from soccer_twos.evaluate import collect_episodes, load_agent

    # Agent construction opens short-lived Unity envs. Keep those ports separate
    # from the actual evaluation env so gRPC has time to release listeners.
    agent1 = load_agent(
        agent1_module_name,
        base_port=offset_base_port(base_port, EVAL_AGENT1_PORT_OFFSET),
    )
    agent2 = load_agent(
        agent2_module_name,
        base_port=offset_base_port(base_port, EVAL_AGENT2_PORT_OFFSET),
    )
    env = soccer_twos.make(
        base_port=offset_base_port(base_port, EVAL_MATCH_PORT_OFFSET),
    )
    try:
        episodes = collect_episodes(env, agent1, agent2, n_episodes)
    finally:
        env.close()

    return summarize_episodes_safe(episodes, agent1_module_name, agent2_module_name)


def evaluate_against_ceia(
    checkpoint_path: str,
    run_name: str,
    local_dir: str,
    eval_episodes: int,
    eval_base_port: int = None,
):
    payload_path = os.path.join(
        local_dir, "evaluations", f"{run_name}_vs_ceia_baseline.json"
    )
    if os.path.isfile(payload_path):
        with open(payload_path) as f:
            cached_payload = json.load(f)
        if (
            cached_payload.get("checkpoint_path") == checkpoint_path
            and cached_payload.get("eval_episodes") == eval_episodes
        ):
            print(f"\n=== Reusing cached evaluation: {run_name} vs ceia_baseline_agent ===")
            print(pretty_print(cached_payload["result"]))
            print(f"[cached] {payload_path}")
            return cached_payload

    previous_checkpoint_path = os.environ.get("SOCCER_CHECKPOINT_PATH")
    previous_agent_name = os.environ.get("SOCCER_AGENT_NAME")
    os.environ["SOCCER_CHECKPOINT_PATH"] = checkpoint_path
    os.environ["SOCCER_AGENT_NAME"] = run_name

    try:
        result = evaluate_matchup(
            CHECKPOINT_AGENT_MODULE,
            CEIA_AGENT_MODULE,
            n_episodes=eval_episodes,
            base_port=eval_base_port,
        )
    finally:
        if previous_checkpoint_path is None:
            os.environ.pop("SOCCER_CHECKPOINT_PATH", None)
        else:
            os.environ["SOCCER_CHECKPOINT_PATH"] = previous_checkpoint_path

        if previous_agent_name is None:
            os.environ.pop("SOCCER_AGENT_NAME", None)
        else:
            os.environ["SOCCER_AGENT_NAME"] = previous_agent_name

    payload = {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "eval_episodes": eval_episodes,
        "agent_module": CHECKPOINT_AGENT_MODULE,
        "baseline_module": CEIA_AGENT_MODULE,
        "result": normalize_for_json(result),
    }

    eval_dir = os.path.dirname(payload_path)
    os.makedirs(eval_dir, exist_ok=True)
    with open(payload_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Evaluation summary ===")
    print(pretty_print(payload["result"]))
    print(f"[saved] {payload_path}")
    return payload


def run_one(
    run_name: str,
    env_name: str,
    env_factory,
    timesteps: int,
    num_workers: int,
    num_gpus: int,
    local_dir: str,
    base_port: Optional[int],
):
    tune.registry.register_env(env_name, env_factory)

    temp_env = open_probe_env(env_factory, base_port)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = build_config(env_name, obs_space, act_space, num_workers, num_gpus, base_port)

    print(f"\n=== Starting run: {run_name} (stop at {timesteps} timesteps) ===")
    analysis = tune.run(
        "PPO",
        name=run_name,
        config=config,
        stop={"timesteps_total": timesteps},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=local_dir,
    )
    trial = analysis.trials[0]
    checkpoint_path = latest_checkpoint_path(trial.logdir)
    print(f"[checkpoint] {run_name}: {checkpoint_path}")
    return {
        "run_name": run_name,
        "trial_dir": trial.logdir,
        "checkpoint_path": checkpoint_path,
        "timesteps_total": timesteps,
        "reused": False,
    }


def get_or_run_one(
    run_name: str,
    env_name: str,
    env_factory,
    timesteps: int,
    num_workers: int,
    num_gpus: int,
    local_dir: str,
    reuse_existing: bool,
    base_port: Optional[int],
):
    if reuse_existing:
        reusable = find_reusable_run(run_name, local_dir, timesteps)
        if reusable is not None:
            print(
                f"\n=== Reusing existing run: {run_name} "
                f"({reusable['timesteps_total']} timesteps) ==="
            )
            print(f"[trial] {reusable['trial_dir']}")
            print(f"[checkpoint] {reusable['checkpoint_path']}")
            reusable["reused"] = True
            return reusable

    return run_one(
        run_name=run_name,
        env_name=env_name,
        env_factory=env_factory,
        timesteps=timesteps,
        num_workers=num_workers,
        num_gpus=num_gpus,
        local_dir=local_dir,
        base_port=base_port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["baseline", "shaped", "both"],
        default="both",
        help="Which run(s) to execute.",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--base-port",
        type=int,
        default=50039,
        help="Base port for SoccerTwos Unity workers during training.",
    )
    parser.add_argument("--local-dir", default=os.path.abspath("./ray_results_compare"))
    parser.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Force a fresh PPO_baseline run instead of reusing an existing one.",
    )
    parser.add_argument(
        "--evaluate-vs-ceia",
        action="store_true",
        help="After training, evaluate each run's latest checkpoint against ceia_baseline_agent.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes when --evaluate-vs-ceia is set.",
    )
    parser.add_argument(
        "--eval-base-port",
        type=int,
        default=None,
        help="Optional base port to pass through to soccer_twos.evaluate.",
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    completed_runs: List[Dict[str, Any]] = []
    baseline_reusable = (
        not args.retrain_baseline
        and find_reusable_run("PPO_baseline", args.local_dir, args.timesteps) is not None
    )
    needs_training = (
        (args.variant in ("baseline", "both") and not baseline_reusable)
        or args.variant in ("shaped", "both")
    )

    # include_dashboard=False avoids the Prometheus/dashboard agent, which tries
    # to bind to the node's hostname and spams `socket.gaierror` when the host
    # can't be resolved (harmless, but noisy).
    if needs_training:
        ray.init(include_dashboard=False)
    try:
        if args.variant in ("baseline", "both"):
            completed_runs.append(
                get_or_run_one(
                    run_name="PPO_baseline",
                    env_name="SoccerBaseline",
                    env_factory=create_rllib_env,
                    timesteps=args.timesteps,
                    num_workers=args.num_workers,
                    num_gpus=args.num_gpus,
                    local_dir=args.local_dir,
                    reuse_existing=not args.retrain_baseline,
                    base_port=args.base_port,
                )
            )
        if args.variant in ("shaped", "both"):
            completed_runs.append(
                get_or_run_one(
                    run_name="PPO_shaped",
                    env_name="SoccerShaped",
                    env_factory=create_rllib_env_shaped,
                    timesteps=args.timesteps,
                    num_workers=args.num_workers,
                    num_gpus=args.num_gpus,
                    local_dir=args.local_dir,
                    reuse_existing=False,
                    base_port=args.base_port,
                )
            )
    finally:
        if needs_training:
            ray.shutdown()

    if args.evaluate_vs_ceia:
        for run in completed_runs:
            evaluate_against_ceia(
                checkpoint_path=run["checkpoint_path"],
                run_name=run["run_name"],
                local_dir=args.local_dir,
                eval_episodes=args.eval_episodes,
                eval_base_port=(
                    args.eval_base_port
                    if args.eval_base_port is not None
                    else args.base_port + 1000
                ),
            )

    print(
        "\nDone. Results saved under "
        f"{args.local_dir}. Run `python plot_compare.py --local-dir {args.local_dir}` "
        "to generate the comparison plot."
    )
