"""
Environment factories and wrappers.

make_env(config, shielded=False)
    Build a highway-fast-v0 environment for training (faster simulation).
    When shielded=True, wrap it in ShieldedEnv.

make_eval_env(shielded=False)
    Build a highway-v0 environment for evaluation (full fidelity).

ShieldedEnv
    Gymnasium wrapper that intercepts env.step(action) and replaces
    the proposed action with a safe one (via LTLSafetyShield) before
    forwarding to the inner environment.

    The wrapper is transparent to SB3: it stores the raw 2-D Kinematics
    observation after each step/reset, passes it to the shield, then
    returns the (possibly modified) transition to SB3.

    Note on Q-learning approximation
    ---------------------------------
    SB3's replay buffer stores (obs, proposed_action, reward, next_obs).
    Here the *executed* action may differ from proposed_action.  This
    introduces a mild off-policy error that is acceptable for a research
    comparison but would need correction (e.g. action masking) for a
    production system.
"""

import gymnasium as gym
import numpy as np
import highway_env  # pyright: ignore[reportMissingImports] — registers highway-v0 etc.  # noqa: F401

from safety_shield import LTLSafetyShield
from config import BASE_ENV_CONFIG, EVAL_ENV_CONFIG, SHIELD_OVERRIDE_PENALTY


def make_env(config: dict | None = None, shielded: bool = False) -> gym.Env:
    """
    Create a configured highway-fast-v0 environment for training.

    Parameters
    ----------
    config   : env config dict (defaults to BASE_ENV_CONFIG)
    shielded : if True, wrap with ShieldedEnv
    """
    cfg = config if config is not None else BASE_ENV_CONFIG
    env = gym.make("highway-fast-v0", config=cfg)
    if shielded:
        env = ShieldedEnv(env)
    return env


def make_eval_env(shielded: bool = False) -> gym.Env:
    """Create a highway-v0 (full-speed) environment for evaluation."""
    env = gym.make("highway-v0", config=EVAL_ENV_CONFIG)
    if shielded:
        env = ShieldedEnv(env)
    return env


class ShieldedEnv(gym.Wrapper):
    """
    Wraps a highway-v0 environment with the LTL safety shield.

    The shield is active both during training (safe exploration) and
    evaluation (hard safety guarantee).  Override statistics are exposed
    via self.shield.override_rate.
    """

    def __init__(self, env: gym.Env, shield: LTLSafetyShield | None = None) -> None:
        super().__init__(env)
        self.shield = shield if shield is not None else LTLSafetyShield()
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: int):
        assert self._last_obs is not None, "call reset() before step()"

        safe_action = self.shield.get_safe_action(self._last_obs, int(action))
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        overridden = safe_action != int(action)
        if overridden:
            reward -= SHIELD_OVERRIDE_PENALTY  # type: ignore

        info["shield_override"] = int(overridden)
        info["proposed_action"] = int(action)
        info["executed_action"] = int(safe_action)
        info["shield_override_rate"] = self.shield.override_rate

        self._last_obs = obs
        return obs, reward, terminated, truncated, info
