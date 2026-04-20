"""
Trains PPO on SoccerTwos twice under identical hyperparameters:
  (a) baseline - raw environment reward
  (b) shaped   - goal-event amplification + optional ball-proximity shaping
                 (see reward_shaping.RewardShapingWrapper)

Writes each run's progress.csv to `./ray_results_compare/PPO_{baseline,shaped}/...`
so that plot_compare.py can overlay the learning curves.

Usage
-----
    python train_compare.py --variant both --timesteps 500000 --num-workers 4
    python train_compare.py --variant baseline
    python train_compare.py --variant shaped
"""

import argparse
import os

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env
from reward_shaping import create_rllib_env_shaped


NUM_ENVS_PER_WORKER = 3


def build_config(env_name: str, obs_space, act_space, num_workers: int, num_gpus: int):
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
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.multiagent_player,
        },
    }


def run_one(
    run_name: str,
    env_name: str,
    env_factory,
    timesteps: int,
    num_workers: int,
    num_gpus: int,
    local_dir: str,
):
    tune.registry.register_env(env_name, env_factory)

    temp_env = env_factory({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = build_config(env_name, obs_space, act_space, num_workers, num_gpus)

    print(f"\n=== Starting run: {run_name} (stop at {timesteps} timesteps) ===")
    tune.run(
        "PPO",
        name=run_name,
        config=config,
        stop={"timesteps_total": timesteps},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=local_dir,
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
    parser.add_argument("--local-dir", default=os.path.abspath("./ray_results_compare"))
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    # include_dashboard=False avoids the Prometheus/dashboard agent, which tries
    # to bind to the node's hostname and spams `socket.gaierror` when the host
    # can't be resolved (harmless, but noisy).
    ray.init(include_dashboard=False)
    try:
        if args.variant in ("baseline", "both"):
            run_one(
                run_name="PPO_baseline",
                env_name="SoccerBaseline",
                env_factory=create_rllib_env,
                timesteps=args.timesteps,
                num_workers=args.num_workers,
                num_gpus=args.num_gpus,
                local_dir=args.local_dir,
            )
        if args.variant in ("shaped", "both"):
            run_one(
                run_name="PPO_shaped",
                env_name="SoccerShaped",
                env_factory=create_rllib_env_shaped,
                timesteps=args.timesteps,
                num_workers=args.num_workers,
                num_gpus=args.num_gpus,
                local_dir=args.local_dir,
            )
    finally:
        ray.shutdown()

    print(
        "\nDone. Results saved under "
        f"{args.local_dir}. Run `python plot_compare.py --local-dir {args.local_dir}` "
        "to generate the comparison plot."
    )
