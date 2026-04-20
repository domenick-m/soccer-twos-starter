"""
Reward-shaping wrapper and env factory used by the comparison training script.

The default SoccerTwos reward (see `soccer_twos.wrappers.MultiAgentUnityWrapper`) is
  r_i = info.reward[i] + info.group_reward[i]
which is dominated by a small per-step time penalty plus a sparse +/-1 goal event.

This wrapper applies two small modifications:
  1. Goal-event amplification. When |r| exceeds a threshold (i.e. a goal was scored
     or conceded) the reward is multiplied by `goal_scale`. The sparse signal
     becomes larger relative to the time penalty, which PPO can exploit sooner.
  2. Ball-proximity dense bonus. When the underlying env exposes per-agent
     player and ball positions in `info` (rollout binary / 345-float obs), we
     subtract `ball_distance_weight * ||player_pos - ball_pos||`. On the default
     training binary (336-float obs) `info` is empty and this term is skipped.
"""

from typing import Any, Dict

import gym
import numpy as np
from ray.rllib import MultiAgentEnv

import soccer_twos


class RewardShapingWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        goal_scale: float = 3.0,
        goal_threshold: float = 0.5,
        ball_distance_weight: float = 0.005,
    ):
        super().__init__(env)
        self.goal_scale = goal_scale
        self.goal_threshold = goal_threshold
        self.ball_distance_weight = ball_distance_weight

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not isinstance(reward, dict):
            return obs, reward, done, info

        shaped: Dict[Any, float] = {}
        for agent_id, r in reward.items():
            if abs(r) >= self.goal_threshold:
                r = r * self.goal_scale

            agent_info = info.get(agent_id, {}) if isinstance(info, dict) else {}
            if (
                isinstance(agent_info, dict)
                and "player_info" in agent_info
                and "ball_info" in agent_info
            ):
                ppos = np.asarray(agent_info["player_info"]["position"], dtype=np.float32)
                bpos = np.asarray(agent_info["ball_info"]["position"], dtype=np.float32)
                r -= self.ball_distance_weight * float(np.linalg.norm(ppos - bpos))

            shaped[agent_id] = r
        return obs, shaped, done, info


class _RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_rllib_env_shaped(env_config: dict = {}):
    """Mirror of `utils.create_rllib_env`, but wraps with `RewardShapingWrapper`."""
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = RewardShapingWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return _RLLibWrapper(env)
