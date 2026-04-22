"""
Configurable reward-shaping wrapper for SoccerTwos.

The base SoccerTwos reward (see `soccer_twos.wrappers.MultiAgentUnityWrapper`) is
    r_i = info.reward[i] + info.group_reward[i]
which is mostly a small per-step time penalty plus sparse +/-1 goal events.

This wrapper keeps the sparse goal signal, then optionally adds dense shaping
terms. Each term has an `enabled` flag and a `weight` so it is easy to tune or
disable without touching the shaping logic.

By default, the wrapper enables:
  - goal-event amplification
  - ball velocity toward the opponent goal
  - player distance to the opponent goal
  - player velocity toward the opponent goal
  - teammate spacing
  - possession balance / passing encouragement

Dense terms require the richer 345-float rollout observation because the env
must expose `player_info` and `ball_info` through `info`. If the env only emits
the default 336-float observation, those terms are skipped automatically.

You can override any defaults by passing `env_config["reward_shaping"]`:

    from reward_shaping import get_default_reward_shaping_config

    shaping = get_default_reward_shaping_config()
    shaping["teammate_spacing"]["enabled"] = False
    shaping["ball_velocity_to_goal"]["weight"] = 0.03
    shaping["possession_balance"]["pass_bonus"] = 0.01
"""

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import gym
import numpy as np
from ray.rllib import MultiAgentEnv

import soccer_twos


TEAM_TO_AGENT_IDS = {
    0: (0, 1),  # blue team
    1: (2, 3),  # orange team
}

AGENT_TO_TEAM = {
    agent_id: team_id
    for team_id, agent_ids in TEAM_TO_AGENT_IDS.items()
    for agent_id in agent_ids
}
TEAM_ATTACK_SIGN = {
    0: 1.0,   # blue attacks toward +x
    1: -1.0,  # orange attacks toward -x
}


DEFAULT_REWARD_SHAPING_CONFIG: Dict[str, Dict[str, Any]] = {
    "goal_event": {
        "enabled": True,
        "scale": 3.0,
        "threshold": 0.5,
    },
    "ball_proximity": {
        "enabled": False,
        "weight": 0.005,
        "distance_scale": 20.0,
    },
    "ball_velocity_to_goal": {
        "enabled": True,
        "weight": 0.02,
        "speed_scale": 10.0,
        "goal_x": 16.0,
        "goal_y": 0.0,
    },
    "goal_distance": {
        "enabled": True,
        "weight": 0.01,
        "distance_scale": 32.0,
        "goal_x": 16.0,
        "goal_y": 0.0,
    },
    "goal_velocity": {
        "enabled": True,
        "weight": 0.01,
        "speed_scale": 10.0,
        "goal_x": 16.0,
        "goal_y": 0.0,
    },
    "teammate_spacing": {
        "enabled": True,
        "weight": 0.01,
        "target_distance": 6.0,
        "tolerance": 4.0,
    },
    "possession_balance": {
        "enabled": True,
        "weight": 0.01,
        "control_radius": 1.75,
        "pass_bonus": 0.02,
    },
}


def get_default_reward_shaping_config() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of the default shaping config for easy external edits."""
    return deepcopy(DEFAULT_REWARD_SHAPING_CONFIG)


def _merge_reward_shaping_config(
    override: Optional[Mapping[str, Mapping[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    config = get_default_reward_shaping_config()
    if override is None:
        return config

    for section, values in override.items():
        if section not in config or not isinstance(values, Mapping):
            config[section] = dict(values) if isinstance(values, Mapping) else values
            continue
        config[section].update(values)
    return config


def _clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _positive_unit_score(error: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (error / scale)))


def _safe_unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return fallback
    return vector / norm


class RewardShapingWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
        goal_scale: Optional[float] = None,
        goal_threshold: Optional[float] = None,
        ball_distance_weight: Optional[float] = None,
    ):
        super().__init__(env)
        self.reward_config = _merge_reward_shaping_config(reward_config)

        # Backwards-compatible overrides for the old wrapper signature.
        if goal_scale is not None:
            self.reward_config["goal_event"]["scale"] = goal_scale
        if goal_threshold is not None:
            self.reward_config["goal_event"]["threshold"] = goal_threshold
        if ball_distance_weight is not None:
            self.reward_config["ball_proximity"]["enabled"] = True
            self.reward_config["ball_proximity"]["weight"] = ball_distance_weight
            self.reward_config["ball_proximity"]["distance_scale"] = 1.0

        self._reset_episode_tracking()

    def reset(self, **kwargs):
        self._reset_episode_tracking()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not isinstance(reward, dict):
            return obs, reward, done, info

        shaped: Dict[Any, float] = {agent_id: float(r) for agent_id, r in reward.items()}

        self._apply_goal_event_scaling(shaped)

        agent_states = self._extract_agent_states(info)
        if not agent_states:
            return obs, shaped, done, info

        self._apply_ball_proximity_reward(shaped, agent_states)
        self._apply_goal_progress_rewards(shaped, agent_states)
        self._apply_teammate_spacing_reward(shaped, agent_states)
        self._apply_ball_velocity_to_goal_reward(shaped, agent_states)
        self._apply_possession_balance_reward(shaped, agent_states)

        return obs, shaped, done, info

    def _reset_episode_tracking(self):
        self._team_possession_steps = {
            team_id: {agent_id: 0 for agent_id in agent_ids}
            for team_id, agent_ids in TEAM_TO_AGENT_IDS.items()
        }
        self._last_controller = None

    def _apply_goal_event_scaling(self, shaped: Dict[Any, float]):
        config = self.reward_config["goal_event"]
        if not config.get("enabled", False):
            return

        threshold = float(config.get("threshold", 0.5))
        scale = float(config.get("scale", 1.0))
        for agent_id, reward_value in shaped.items():
            if abs(reward_value) >= threshold:
                shaped[agent_id] = reward_value * scale

    def _extract_agent_states(self, info: Any) -> Dict[int, Dict[str, np.ndarray]]:
        if not isinstance(info, dict):
            return {}

        states: Dict[int, Dict[str, np.ndarray]] = {}
        for agent_id, agent_info in info.items():
            if (
                not isinstance(agent_id, int)
                or not isinstance(agent_info, dict)
                or "player_info" not in agent_info
                or "ball_info" not in agent_info
            ):
                continue

            player_info = agent_info["player_info"]
            ball_info = agent_info["ball_info"]
            player_position = player_info.get("position")
            player_velocity = player_info.get("velocity")
            ball_position = ball_info.get("position")
            ball_velocity = ball_info.get("velocity")
            if (
                player_position is None
                or player_velocity is None
                or ball_position is None
                or ball_velocity is None
            ):
                continue

            states[agent_id] = {
                "player_position": np.asarray(player_position, dtype=np.float32),
                "player_velocity": np.asarray(player_velocity, dtype=np.float32),
                "ball_position": np.asarray(ball_position, dtype=np.float32),
                "ball_velocity": np.asarray(ball_velocity, dtype=np.float32),
            }
        return states

    def _goal_center(self, team_id: int, config: Mapping[str, Any]) -> np.ndarray:
        goal_x = float(config.get("goal_x", 16.0)) * TEAM_ATTACK_SIGN[team_id]
        goal_y = float(config.get("goal_y", 0.0))
        return np.asarray([goal_x, goal_y], dtype=np.float32)

    def _apply_ball_proximity_reward(
        self,
        shaped: Dict[Any, float],
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ):
        config = self.reward_config["ball_proximity"]
        if not config.get("enabled", False):
            return

        weight = float(config.get("weight", 0.0))
        distance_scale = float(config.get("distance_scale", 1.0))
        if weight == 0.0 or distance_scale <= 0:
            return

        for agent_id, state in agent_states.items():
            distance = float(
                np.linalg.norm(state["player_position"] - state["ball_position"])
            )
            shaped[agent_id] -= weight * min(distance / distance_scale, 1.0)

    def _apply_goal_progress_rewards(
        self,
        shaped: Dict[Any, float],
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ):
        distance_config = self.reward_config["goal_distance"]
        velocity_config = self.reward_config["goal_velocity"]
        distance_enabled = distance_config.get("enabled", False)
        velocity_enabled = velocity_config.get("enabled", False)
        if not distance_enabled and not velocity_enabled:
            return

        for agent_id, state in agent_states.items():
            team_id = AGENT_TO_TEAM.get(agent_id)
            if team_id is None:
                continue

            player_position = state["player_position"]
            player_velocity = state["player_velocity"]

            if distance_enabled:
                goal_center = self._goal_center(team_id, distance_config)
                distance = float(np.linalg.norm(goal_center - player_position))
                distance_score = _positive_unit_score(
                    distance,
                    float(distance_config.get("distance_scale", 32.0)),
                )
                shaped[agent_id] += float(distance_config.get("weight", 0.0)) * distance_score

            if velocity_enabled:
                goal_center = self._goal_center(team_id, velocity_config)
                fallback = np.asarray([TEAM_ATTACK_SIGN[team_id], 0.0], dtype=np.float32)
                direction = _safe_unit_vector(goal_center - player_position, fallback)
                speed_scale = float(velocity_config.get("speed_scale", 10.0))
                if speed_scale > 0:
                    toward_goal = float(np.dot(player_velocity, direction)) / speed_scale
                    shaped[agent_id] += (
                        float(velocity_config.get("weight", 0.0))
                        * _clip_unit(toward_goal)
                    )

    def _apply_teammate_spacing_reward(
        self,
        shaped: Dict[Any, float],
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ):
        config = self.reward_config["teammate_spacing"]
        if not config.get("enabled", False):
            return

        weight = float(config.get("weight", 0.0))
        target_distance = float(config.get("target_distance", 6.0))
        tolerance = float(config.get("tolerance", 4.0))
        if weight == 0.0 or tolerance <= 0:
            return

        for agent_ids in TEAM_TO_AGENT_IDS.values():
            if not all(agent_id in agent_states for agent_id in agent_ids):
                continue

            player_a = agent_states[agent_ids[0]]["player_position"]
            player_b = agent_states[agent_ids[1]]["player_position"]
            actual_distance = float(np.linalg.norm(player_a - player_b))
            spacing_score = _positive_unit_score(
                abs(actual_distance - target_distance),
                tolerance,
            )
            for agent_id in agent_ids:
                shaped[agent_id] += weight * spacing_score

    def _shared_ball_state(
        self,
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ) -> Optional[Dict[str, np.ndarray]]:
        for state in agent_states.values():
            return {
                "ball_position": state["ball_position"],
                "ball_velocity": state["ball_velocity"],
            }
        return None

    def _apply_ball_velocity_to_goal_reward(
        self,
        shaped: Dict[Any, float],
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ):
        config = self.reward_config["ball_velocity_to_goal"]
        if not config.get("enabled", False):
            return

        weight = float(config.get("weight", 0.0))
        speed_scale = float(config.get("speed_scale", 10.0))
        shared_ball_state = self._shared_ball_state(agent_states)
        if weight == 0.0 or speed_scale <= 0 or shared_ball_state is None:
            return

        ball_position = shared_ball_state["ball_position"]
        ball_velocity = shared_ball_state["ball_velocity"]
        for team_id, agent_ids in TEAM_TO_AGENT_IDS.items():
            goal_center = self._goal_center(team_id, config)
            fallback = np.asarray([TEAM_ATTACK_SIGN[team_id], 0.0], dtype=np.float32)
            direction = _safe_unit_vector(goal_center - ball_position, fallback)
            toward_goal = float(np.dot(ball_velocity, direction)) / speed_scale
            team_bonus = weight * _clip_unit(toward_goal)
            for agent_id in agent_ids:
                if agent_id in shaped:
                    shaped[agent_id] += team_bonus

    def _infer_controller(
        self,
        agent_states: Dict[int, Dict[str, np.ndarray]],
        control_radius: float,
    ) -> Optional[int]:
        if control_radius <= 0:
            return None

        best_agent = None
        best_distance = float("inf")
        for agent_id, state in agent_states.items():
            distance = float(
                np.linalg.norm(state["player_position"] - state["ball_position"])
            )
            if distance <= control_radius and distance < best_distance:
                best_agent = agent_id
                best_distance = distance
        return best_agent

    def _apply_possession_balance_reward(
        self,
        shaped: Dict[Any, float],
        agent_states: Dict[int, Dict[str, np.ndarray]],
    ):
        config = self.reward_config["possession_balance"]
        if not config.get("enabled", False):
            return

        controller = self._infer_controller(
            agent_states,
            float(config.get("control_radius", 1.75)),
        )
        if controller is None:
            return

        team_id = AGENT_TO_TEAM.get(controller)
        if team_id is None:
            return

        self._team_possession_steps[team_id][controller] += 1
        team_counts = self._team_possession_steps[team_id]
        total_steps = float(sum(team_counts.values()))
        if total_steps <= 0:
            return

        count_values = list(team_counts.values())
        balance_score = 1.0 - (abs(count_values[0] - count_values[1]) / total_steps)
        team_bonus = float(config.get("weight", 0.0)) * balance_score

        pass_bonus = float(config.get("pass_bonus", 0.0))
        if (
            pass_bonus > 0.0
            and self._last_controller is not None
            and self._last_controller != controller
            and AGENT_TO_TEAM.get(self._last_controller) == team_id
        ):
            team_bonus += pass_bonus

        for agent_id in TEAM_TO_AGENT_IDS[team_id]:
            if agent_id in shaped:
                shaped[agent_id] += team_bonus

        self._last_controller = controller


class _RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_rllib_env_shaped(env_config: Optional[Mapping[str, Any]] = None):
    """
    Mirror of `utils.create_rllib_env`, but wraps the env with reward shaping.

    Any `env_config["reward_shaping"]` mapping is merged into
    `DEFAULT_REWARD_SHAPING_CONFIG`.
    """
    env_config = {} if env_config is None else env_config
    shaping_config = env_config.get("reward_shaping")
    env_kwargs = dict(env_config)
    if hasattr(env_config, "worker_index"):
        env_kwargs["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env_kwargs.pop("reward_shaping", None)

    env = soccer_twos.make(**env_kwargs)
    env = RewardShapingWrapper(env, reward_config=shaping_config)
    if env_config.get("multiagent") is False:
        return env
    return _RLLibWrapper(env)
