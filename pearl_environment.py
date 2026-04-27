"""
Pearl-compatible environment wrappers for highway-env.

HighwayPearlEnv
    Thin wrapper around a Gymnasium highway-env exposing Pearl's Environment
    interface.  Observation is flattened to a 1-D float32 tensor; every
    call to reset/step returns the full 5-action space.

ShieldedHighwayPearlEnv
    Extends HighwayPearlEnv to integrate the LTL safety shield as action
    masking.  After every reset/step the shield evaluates which of the 5
    actions satisfy φ₁–φ₄ and returns them as `available_action_space`.

    This is the key correctness improvement over the SB3 approach:
    • Pearl's DeepQLearning selects from the safe subset → no off-policy error.
    • The Bellman backup uses the safe next-action set stored in the replay
      buffer → target Q-values are computed over only safe next actions.
"""

import numpy as np
import torch
import gymnasium as gym
import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401

from pearl.api.environment import Environment
from pearl.api.action_result import ActionResult
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from safety_shield import LTLSafetyShield
from config import (
    BASE_ENV_CONFIG,
    EVAL_ENV_CONFIG,
    SHIELD_OVERRIDE_PENALTY,
)


N_ACTIONS = 5


def _flatten(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs.flatten(), dtype=torch.float32)


def _full_action_space() -> DiscreteActionSpace:
    return DiscreteActionSpace([torch.tensor([i]) for i in range(N_ACTIONS)])


def make_pearl_env(
    config: dict | None = None,
    shielded: bool = False,
    fast: bool = True,
) -> "HighwayPearlEnv":
    """Create a Pearl-compatible highway training environment."""
    cfg = config if config is not None else BASE_ENV_CONFIG
    gym_id = "highway-fast-v0" if fast else "highway-v0"
    gym_env = gym.make(gym_id, config=cfg)
    if shielded:
        return ShieldedHighwayPearlEnv(gym_env)
    return HighwayPearlEnv(gym_env)


def make_pearl_eval_env(shielded: bool = False) -> "HighwayPearlEnv":
    """Create a Pearl-compatible evaluation environment (full-fidelity)."""
    gym_env = gym.make("highway-v0", config=EVAL_ENV_CONFIG)
    if shielded:
        return ShieldedHighwayPearlEnv(gym_env)
    return HighwayPearlEnv(gym_env)


class HighwayPearlEnv(Environment):
    """Pearl Environment wrapping a Gymnasium highway-env.

    Observation: flat float32 tensor of shape (N_OBS_VEHICLES * N_FEATURES,)
    Action:      torch.Tensor of shape (1,) with value in {0,1,2,3,4}
    """

    def __init__(self, gym_env: gym.Env) -> None:
        self._env = gym_env

    def reset(self) -> tuple[torch.Tensor, DiscreteActionSpace]:  # pyright: ignore[reportIncompatibleMethodOverride]
        obs, _info = self._env.reset()
        return _flatten(obs), _full_action_space()

    def step(self, action) -> ActionResult:
        obs, reward, terminated, truncated, info = self._env.step(int(action.item()))
        return ActionResult(
            observation=_flatten(obs),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info or {},
            available_action_space=_full_action_space(),
        )

    @property
    def action_space(self) -> DiscreteActionSpace:
        return _full_action_space()

    @property
    def observation_space(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._env.observation_space

    def close(self) -> None:
        self._env.close()


class ShieldedHighwayPearlEnv(HighwayPearlEnv):
    """HighwayPearlEnv with LTL safety shield integrated as action masking.

    At every transition the shield computes which actions satisfy φ₁–φ₄ and
    returns them as `available_action_space`.  Pearl stores these per
    transition in the replay buffer, so the Bellman target:

        r  +  γ · max_{a' ∈ safe(s')} Q(s', a')

    is computed over only the safe actions for the next state.

    Statistics
    ----------
    filter_rate : fraction of steps where at least one action was filtered.
    """

    def __init__(self, gym_env: gym.Env, shield: LTLSafetyShield | None = None) -> None:
        super().__init__(gym_env)
        self.shield = shield if shield is not None else LTLSafetyShield()
        self._total_steps: int = 0
        self._filtered_steps: int = 0

    def reset(self) -> tuple[torch.Tensor, DiscreteActionSpace]:
        obs, _info = self._env.reset()
        return _flatten(obs), self._safe_action_space(obs)

    def step(self, action) -> ActionResult:
        obs, reward, terminated, truncated, info = self._env.step(int(action.item()))
        avail = self._safe_action_space(obs)
        info = info or {}
        info["shield_filter_rate"] = self.filter_rate
        info["n_safe_actions"] = len(avail.actions)

        # Penalise proportionally to how constrained the next state is.
        # Recreates the spirit of the SB3 SHIELD_OVERRIDE_PENALTY: the DQN
        # learns to proactively maintain spacing so all actions stay available,
        # rather than stalling in traffic where only SLOWER is safe.
        n_restricted = N_ACTIONS - len(avail.actions)
        if n_restricted > 0:
            reward -= SHIELD_OVERRIDE_PENALTY * (n_restricted / N_ACTIONS)  # type: ignore

        return ActionResult(
            observation=_flatten(obs),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
            available_action_space=avail,
        )

    def _safe_action_space(self, obs: np.ndarray) -> DiscreteActionSpace:
        self._total_steps += 1
        safe = [
            torch.tensor([a])
            for a in range(N_ACTIONS)
            if self.shield.is_action_safe(obs, a)
        ]
        if not safe:
            safe = [torch.tensor([4])]  # SLOWER is always the safe fallback
        if len(safe) < N_ACTIONS:
            self._filtered_steps += 1
        return DiscreteActionSpace(safe)

    @property
    def filter_rate(self) -> float:
        """Fraction of steps where at least one action was filtered."""
        return self._filtered_steps / max(self._total_steps, 1)
