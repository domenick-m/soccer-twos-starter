"""Goal-aware potential-based reward shaping for soccer_twos.

Design principle: the only form of state-dependent reward shaping that is
guaranteed to preserve the optimal policy of the original MDP is
potential-based shaping (Ng, Harada, Russell 1999):

    F(s, s') = gamma * Phi(s') - Phi(s)

We therefore put every continuous shaping signal inside a single scalar
potential Phi, then let the PBRS telescoping structure induce the per-step
reward. Only genuinely event-like signals (a discrete kick) are added outside
the potential.

Potential (per agent i, derived only from that agent's raw 42x8 ray cast):

    Phi_i(s) = -alpha * d_ball(i)  -  beta * d_opp_goal(i)

  - d_ball(i)     : agent-i's min ray-distance to the ball (1.0 when unseen)
  - d_opp_goal(i) : agent-i's min ray-distance to the opposing goal
  - alpha dominates (default 1.0). beta is smaller (default 0.3) — enough to
    encourage upfield positioning without overriding ball pursuit.

Why couple to the opponent goal? A ball-only potential pulls both team-mates
to the ball. Adding -beta * d_opp_goal gives the off-ball agent a gradient
toward the opponent's half, implicitly producing spread (no ad-hoc threshold
needed) and creating passing lanes.

Event term: kick_bonus(i).  PBRS on Phi = -d_ball has an unwanted side
effect — when an agent kicks the ball, d_ball jumps up, so Phi drops, so
gamma*Phi(s') - Phi(s) is negative. Without a counter-signal the agent is
trained to stay glued to the ball and never kick it away. The kick_bonus is a
small direct reward gated on a genuine kick event (agent was touching the
ball, ball is now away):

    kick_bonus(i) = base + extra * max(0, 1 - d_opp_goal(i))

The extra term makes the bonus direction-aware: a kick taken near the
opponent goal (a probable shot) is worth more than one taken near the agent's
own half (a clearance). This exploits information available from the rays
(seeing the opponent goal) without needing ball-velocity geometry.
"""

from typing import Dict

import gym
import numpy as np
from ray.rllib import MultiAgentEnv


# Ray channel layout per the soccer_twos RayPerceptionSensor:
#   [ball, blue_goal, purple_goal, blue_agent, purple_agent, wall, has_hit, distance]
_RAY_WIDTH = 8
_NUM_RAYS = 42
_CH_BALL = 0
_CH_BLUE_GOAL = 1
_CH_PURPLE_GOAL = 2
_CH_DIST = 7

# Contact detection (kick events)
_CONTACT_RADIUS = 0.15
_KICK_GAP = 0.10


def _opp_goal_channel(agent_id: int) -> int:
    # Blue team = agent ids 0, 1 (attacks purple goal).
    # Purple team = agent ids 2, 3 (attacks blue goal).
    return _CH_PURPLE_GOAL if agent_id < 2 else _CH_BLUE_GOAL


def _closest_tagged_ray(obs: np.ndarray, tag_channel: int) -> float:
    """Minimum normalized distance across the 42 rays where tag_channel fires."""
    closest = 1.0
    for ray in range(_NUM_RAYS):
        offset = ray * _RAY_WIDTH
        if obs[offset + tag_channel] > 0.5:
            d = float(obs[offset + _CH_DIST])
            if d < closest:
                closest = d
    return closest


class GoalAwarePBRSWrapper(gym.core.Wrapper, MultiAgentEnv):
    """Per-agent PBRS with a direction-aware kick event bonus.

    Hyperparameters that matter:
      - potential_scale : overall magnitude of the PBRS term. Keep small
        (default 0.01) so PBRS adds dense gradient without drowning the
        sparse +/-1 goal signal.
      - pbrs_gamma      : gamma used *inside* PBRS. Should match the PPO
        discount factor for policy-invariance to hold exactly.
    """

    def __init__(
        self,
        env,
        alpha: float = 1.0,
        beta: float = 0.3,
        pbrs_gamma: float = 0.99,
        potential_scale: float = 0.01,
        kick_base: float = 0.04,
        kick_goal_bonus: float = 0.06,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.pbrs_gamma = pbrs_gamma
        self.potential_scale = potential_scale
        self.kick_base = kick_base
        self.kick_goal_bonus = kick_goal_bonus

        self._last_phi: Dict[int, float] = {}
        self._last_d_ball: Dict[int, float] = {}

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _compute_phi(self, agent_obs: np.ndarray, agent_id: int):
        d_ball = _closest_tagged_ray(agent_obs, _CH_BALL)
        d_opp_goal = _closest_tagged_ray(agent_obs, _opp_goal_channel(agent_id))
        phi = -self.alpha * d_ball - self.beta * d_opp_goal
        return phi, d_ball, d_opp_goal

    def _augment_rewards(
        self,
        obs: Dict[int, np.ndarray],
        base_rewards: Dict[int, float],
    ) -> Dict[int, float]:
        augmented = {}
        for agent_id, agent_obs in obs.items():
            phi_now, d_ball_now, d_opp_goal_now = self._compute_phi(agent_obs, agent_id)

            pbrs_term = 0.0
            if agent_id in self._last_phi:
                phi_prev = self._last_phi[agent_id]
                pbrs_term = self.potential_scale * (
                    self.pbrs_gamma * phi_now - phi_prev
                )
            self._last_phi[agent_id] = phi_now

            kick_term = 0.0
            d_ball_prev = self._last_d_ball.get(agent_id, 1.0)
            was_touching = d_ball_prev < _CONTACT_RADIUS
            ball_escaped = (d_ball_now - d_ball_prev) > _KICK_GAP
            if was_touching and ball_escaped:
                proximity_to_opp = max(0.0, 1.0 - d_opp_goal_now)
                kick_term = self.kick_base + self.kick_goal_bonus * proximity_to_opp
            self._last_d_ball[agent_id] = d_ball_now

            augmented[agent_id] = (
                float(base_rewards.get(agent_id, 0.0)) + pbrs_term + kick_term
            )
        return augmented

    def reset(self, **kwargs):
        self._last_phi.clear()
        self._last_d_ball.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)
        if isinstance(rewards, dict) and isinstance(obs, dict):
            rewards = self._augment_rewards(obs, rewards)
        return obs, rewards, dones, infos


# ── RLLib environment factory ────────────────────────────────────────────────

class _RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_rllib_env_pbrs(env_config=None):
    """Create a SoccerTwos env wrapped with GoalAwarePBRSWrapper.

    env_config keys consumed:
      - alpha, beta, pbrs_gamma, potential_scale, kick_base, kick_goal_bonus
        (all optional, forwarded to GoalAwarePBRSWrapper)
      - All other keys forwarded to soccer_twos.make()
    """
    import time
    from mlagents_envs.exception import UnityWorkerInUseException, UnityTimeOutException
    import soccer_twos

    env_config = {} if env_config is None else env_config

    # Extract PBRS hyperparams before passing to soccer_twos.make
    pbrs_kwargs = {}
    env_kwargs = dict(env_config)
    for key in ("alpha", "beta", "pbrs_gamma", "potential_scale",
                "kick_base", "kick_goal_bonus"):
        if key in env_kwargs:
            pbrs_kwargs[key] = env_kwargs.pop(key)

    # Worker ID for port assignment — worker_index is set by RLLib as an
    # attribute on the env_config object, not as a dict key.
    if hasattr(env_config, "worker_index"):
        env_kwargs["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    # Remove internal keys not understood by soccer_twos.make
    for key in list(env_kwargs.keys()):
        if isinstance(key, str) and key.startswith("_"):
            env_kwargs.pop(key, None)

    max_retries = 50
    env = None
    for attempt in range(max_retries):
        try:
            env = soccer_twos.make(**env_kwargs)
            break
        except (UnityWorkerInUseException, UnityTimeOutException) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0 + attempt * 0.5)

    if env is None:
        raise RuntimeError("Could not create SoccerTwos environment")

    env = GoalAwarePBRSWrapper(env, **pbrs_kwargs)
    env = _RLLibWrapper(env)
    return env
