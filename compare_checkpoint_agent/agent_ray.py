import os
import pickle
from typing import Any, Dict, Optional

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = os.environ.get("SOCCER_CHECKPOINT_ALGORITHM", "PPO")
POLICY_NAME = os.environ.get("SOCCER_CHECKPOINT_POLICY_NAME", "default")
CHECKPOINT_ENV_VAR = "SOCCER_CHECKPOINT_PATH"


def _resolve_checkpoint_path() -> str:
    checkpoint_path = os.environ.get(CHECKPOINT_ENV_VAR)
    if not checkpoint_path:
        raise ValueError(
            f"Missing required environment variable {CHECKPOINT_ENV_VAR}."
        )

    checkpoint_path = os.path.abspath(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if os.path.isdir(checkpoint_path):
        candidates = [
            os.path.join(checkpoint_path, name)
            for name in os.listdir(checkpoint_path)
            if name.startswith("checkpoint-")
        ]
        if len(candidates) == 1 and os.path.isfile(candidates[0]):
            return candidates[0]

    raise ValueError(f"Checkpoint path does not resolve to a checkpoint file: {checkpoint_path}")


def _maybe_unpickle(value: Any) -> Any:
    if isinstance(value, bytes):
        return pickle.loads(value)
    return value


def _find_policy_state(container: Any, policy_name: str) -> Optional[Dict[str, Any]]:
    container = _maybe_unpickle(container)
    if not isinstance(container, dict):
        return None

    direct = container.get(policy_name)
    if isinstance(direct, dict):
        return direct

    for key in ("worker", "state", "policy_states"):
        nested = container.get(key)
        if nested is None:
            continue
        policy_state = _find_policy_state(nested, policy_name)
        if policy_state is not None:
            return policy_state

    return None


def _extract_policy_weights(checkpoint_path: str, policy_name: str) -> Dict[str, np.ndarray]:
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    policy_state = _find_policy_state(checkpoint_data, policy_name)
    if policy_state is None:
        top_keys = list(checkpoint_data.keys()) if isinstance(checkpoint_data, dict) else None
        raise ValueError(
            f"Policy {policy_name!r} not found in checkpoint {checkpoint_path}. "
            f"Top-level keys: {top_keys}"
        )

    if "weights" in policy_state and isinstance(policy_state["weights"], dict):
        weights = policy_state["weights"]
    else:
        weights = {
            key: value
            for key, value in policy_state.items()
            if key != "_optimizer_variables" and isinstance(value, np.ndarray)
        }

    if not weights:
        raise ValueError(
            f"No model weights found for policy {policy_name!r} in checkpoint {checkpoint_path}."
        )
    return weights


class CheckpointAgent(AgentInterface):
    """
    Loads an RLlib policy from a checkpoint specified via environment variable.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True, include_dashboard=False)

        checkpoint_path = _resolve_checkpoint_path()
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or its parent directory."
            )

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        self.policy = agent.get_policy(POLICY_NAME)
        self.policy.set_weights(_extract_policy_weights(checkpoint_path, POLICY_NAME))
        self.name = os.environ.get("SOCCER_AGENT_NAME", "CheckpointAgent")

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
