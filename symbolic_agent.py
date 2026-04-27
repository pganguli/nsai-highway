"""
Pure-symbolic agent for highway driving.

Implements a finite-state machine (FSM) whose transitions are guarded by
predicates derived from the Kinematics observation.  The design is inspired
by two well-known microscopic traffic models:

  IDM  (Intelligent Driver Model) — longitudinal car-following
  MOBIL (Minimizing Overall Braking Induced by Lane changes) — lane selection

States
------
  CRUISE       Front is clear → accelerate / maintain speed.
  FOLLOW       Front vehicle too close but no lane change possible → decelerate.
  OVERTAKE_L   Left lane is clear and faster progress is possible → change left.
  KEEP_RIGHT   Not in rightmost lane, right is clear, front is clear → keep-right
               rule (highway-env rewards right-lane occupancy).

Transition guard predicates (same LTL-flavored semantics as the shield):
  front_clear(obs)       ¬ close_vehicle_ahead
  left_clear(obs)        ¬ obstacle_in_left_lane
  right_clear(obs)       ¬ obstacle_in_right_lane

The agent exposes a predict() method matching the SB3 Policy interface so it
can be evaluated with the same harness as the trained DQN models.
"""

import numpy as np
from enum import Enum, auto

from config import (
    X_SCALE,
    VX_SCALE,
    LANE_WIDTH_NORM,
    LANE_HALF_WIDTH,
    SHIELD_FRONT_DIST,
    SHIELD_TTC_SEC,
    SHIELD_SIDE_MARGIN,
)

# Discrete action indices (DiscreteMetaAction)
A_LANE_LEFT = 0
A_IDLE = 1
A_LANE_RIGHT = 2
A_FASTER = 3
A_SLOWER = 4


class FSMState(Enum):
    CRUISE = auto()
    FOLLOW = auto()
    OVERTAKE_L = auto()
    KEEP_RIGHT = auto()


class SymbolicAgent:
    """
    Rule-based FSM agent.  No training needed.

    The agent is deliberately conservative: safety always takes priority over
    speed.  This makes it a useful lower bound on the neurosymbolic agent,
    which should achieve similar safety with higher average speed thanks to
    the learned neural component.
    """

    # slightly tighter thresholds than the shield so the agent
    # acts proactively before the shield would need to intervene
    FRONT_DIST = SHIELD_FRONT_DIST * 1.2  # 24 m
    TTC_SEC = SHIELD_TTC_SEC * 1.2  # 2.4 s
    SIDE_MARGIN = SHIELD_SIDE_MARGIN * 1.1  # 22 m

    def __init__(self) -> None:
        self.state = FSMState.CRUISE

    # ── observation predicates ────────────────────────────────────────────────

    def _front_clear(self, others: np.ndarray) -> bool:
        """True when no vehicle ahead in the current lane is dangerously close."""
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy, dvx = float(v[1]), float(v[2]), float(v[3])
            if dx > 0 and abs(dy) < LANE_HALF_WIDTH:
                if dx < self.FRONT_DIST:
                    return False
                if dvx < -1e-3:
                    ttc = (dx * X_SCALE) / (abs(dvx) * VX_SCALE)
                    if ttc < self.TTC_SEC:
                        return False
        return True

    def _left_clear(self, others: np.ndarray) -> bool:
        """True when the left adjacent lane is clear for a lane change."""
        target_dy = -LANE_WIDTH_NORM
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy = float(v[1]), float(v[2])
            if abs(dy - target_dy) < LANE_HALF_WIDTH * 1.5:
                if abs(dx) < self.SIDE_MARGIN:
                    return False
        return True

    def _right_clear(self, others: np.ndarray) -> bool:
        """True when the right adjacent lane is clear for a lane change."""
        target_dy = LANE_WIDTH_NORM
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy = float(v[1]), float(v[2])
            if abs(dy - target_dy) < LANE_HALF_WIDTH * 1.5:
                if abs(dx) < self.SIDE_MARGIN:
                    return False
        return True

    def _ego_in_rightmost_lane(self, others: np.ndarray) -> bool:
        """
        Heuristic: if no observed vehicle has dy > +LANE_WIDTH_NORM * 0.5,
        ego is probably in the rightmost lane (no right lane exists).
        """
        for v in others:
            if v[0] < 0.5:
                continue
            dy = float(v[2])
            if dy > LANE_WIDTH_NORM * 0.5:
                return False
        return True

    # ── FSM transition + action selection ────────────────────────────────────

    def _transition(self, others: np.ndarray) -> FSMState:
        fc = self._front_clear(others)
        lc = self._left_clear(others)
        rc = self._right_clear(others)
        rm = self._ego_in_rightmost_lane(others)

        if not fc:
            if lc:
                return FSMState.OVERTAKE_L
            return FSMState.FOLLOW

        # front is clear
        if not rm and rc:
            return FSMState.KEEP_RIGHT

        return FSMState.CRUISE

    def _action_for_state(self, state: FSMState) -> int:
        return {
            FSMState.CRUISE: A_FASTER,
            FSMState.FOLLOW: A_SLOWER,
            FSMState.OVERTAKE_L: A_LANE_LEFT,
            FSMState.KEEP_RIGHT: A_LANE_RIGHT,
        }[state]

    # ── public interface ──────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Compute action from observation.

        Matches the SB3 Model.predict() signature:
          returns (action, state)  where state=None for non-recurrent agents.

        obs : shape (N_vehicles, N_features)
              OR flattened (N_vehicles * N_features,)
              OR VecEnv batch (n_envs, N_vehicles, N_features)
        """
        from config import N_OBS_VEHICLES, N_FEATURES

        if obs.ndim == 1:
            obs = obs.reshape(N_OBS_VEHICLES, N_FEATURES)
        elif obs.ndim == 3:
            # VecEnv batch — take the first env
            obs = obs[0]

        others = obs[1:]
        self.state = self._transition(others)
        action = self._action_for_state(self.state)
        return action, None

    def reset(self) -> None:
        self.state = FSMState.CRUISE
