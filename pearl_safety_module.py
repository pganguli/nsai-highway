"""
Pearl SafetyModule wrapping the LTL hard safety shield.

LTLShieldSafetyModule implements Pearl's SafetyModule ABC:

    filter_action(subjective_state, action_space) → filtered ActionSpace

The module is used by PearlAgent.act() to restrict action selection to the
safe subset before any exploration or exploitation decision.  When combined
with ShieldedHighwayPearlEnv (which also provides filtered available actions
in each ActionResult), the two are consistent:
  • the env guarantees correct Bellman targets (next-state safe action set)
  • this module guarantees safe action selection at inference time
  • both use the same LTLSafetyShield predicate logic → no contradictions
"""

import numpy as np
import torch
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from safety_shield import LTLSafetyShield
from config import N_OBS_VEHICLES, N_FEATURES
from pearl_environment import N_ACTIONS


class LTLShieldSafetyModule(SafetyModule):
    """
    Pearl SafetyModule backed by the LTL symbolic safety shield.

    Implements hard constraint safety — no learning is required.  Each call
    to filter_action() reshapes the flat observation tensor to 2-D, evaluates
    φ₁–φ₄ for every candidate action, and returns only the safe subset as a
    DiscreteActionSpace.

    Parameters
    ----------
    shield : LTLSafetyShield, optional
        If provided, the shield is shared with the environment wrapper so
        statistics and thresholds stay in sync.  If None, a fresh shield
        is created.
    """

    def __init__(self, shield: LTLSafetyShield | None = None) -> None:
        super().__init__()
        self._shield = shield if shield is not None else LTLSafetyShield()

    # ── SafetyModule ABC ──────────────────────────────────────────────────────

    def filter_action(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
    ) -> ActionSpace:
        """Return the subset of action_space satisfying φ₁–φ₄."""
        if subjective_state is None:
            return action_space

        obs_2d: np.ndarray = (
            subjective_state.detach().cpu().numpy().reshape(N_OBS_VEHICLES, N_FEATURES)
        )

        safe = [
            a
            for a in action_space.actions
            if self._shield.is_action_safe(obs_2d, int(a.item()))
        ]

        if not safe:
            # Priority fallback: SLOWER > IDLE > LANE_LEFT > LANE_RIGHT
            for fb in (4, 1, 0, 2):
                if self._shield.is_action_safe(obs_2d, fb):
                    safe = [torch.tensor([fb])]
                    break
            else:
                safe = [torch.tensor([4])]  # SLOWER — unconditionally safe

        return DiscreteActionSpace(safe)

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass  # hard constraints need no learning

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    def compare(self, other: "SafetyModule") -> str:
        if not isinstance(other, LTLShieldSafetyModule):
            return f"other is {type(other).__name__}, expected LTLShieldSafetyModule"
        return ""
