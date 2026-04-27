"""
LTL-inspired safety shield for highway driving.

The shield enforces four safety invariants (LTL "G" = "globally always"):

  φ₁  G( dist_front > FRONT_DIST_MIN )
        "Never tailgate: always keep a minimum following distance."

  φ₂  G( ttc_front  > TTC_MIN )
        "Always maintain a safe time-to-collision with the car ahead."

  φ₃  G( action=LANE_LEFT  →  ¬obstacle_in_left_lane )
        "Only change to the left lane when it is clear."

  φ₄  G( action=LANE_RIGHT →  ¬obstacle_in_right_lane )
        "Only change to the right lane when it is clear."

  Safety  =  φ₁ ∧ φ₂ ∧ φ₃ ∧ φ₄

At each step the shield checks whether the neural policy's proposed action
satisfies Safety.  If it does not, the shield substitutes a fallback action
chosen from a priority list: SLOWER > IDLE > LANE_LEFT > LANE_RIGHT.

Usage
-----
  shield = LTLSafetyShield()
  safe_action = shield.get_safe_action(obs_2d, proposed_action)
"""

import numpy as np
from enum import IntEnum

from config import (
    X_SCALE, VX_SCALE,
    LANE_WIDTH_NORM, LANE_HALF_WIDTH,
    SHIELD_FRONT_DIST, SHIELD_TTC_SEC, SHIELD_SIDE_MARGIN,
)


class Action(IntEnum):
    LANE_LEFT  = 0
    IDLE       = 1
    LANE_RIGHT = 2
    FASTER     = 3
    SLOWER     = 4


class LTLSafetyShield:
    """
    Symbolic safety monitor that enforces φ₁–φ₄.

    Parameters
    ----------
    front_dist  : normalized minimum following distance  (φ₁)
    ttc_sec     : minimum time-to-collision in seconds    (φ₂)
    side_margin : normalized longitudinal clearance for   (φ₃/φ₄)
                  a safe lane change
    """

    def __init__(
        self,
        front_dist: float = SHIELD_FRONT_DIST,
        ttc_sec: float    = SHIELD_TTC_SEC,
        side_margin: float = SHIELD_SIDE_MARGIN,
    ) -> None:
        self.front_dist  = front_dist
        self.ttc_sec     = ttc_sec
        self.side_margin = side_margin

        # statistics
        self.total_steps    = 0
        self.override_steps = 0

    # ── LTL predicates ────────────────────────────────────────────────────────

    def _phi1_front_distance(self, others: np.ndarray) -> bool:
        """φ₁: no vehicle in the current lane is closer than front_dist ahead."""
        for v in others:
            if v[0] < 0.5:          # vehicle not present
                continue
            dx, dy = float(v[1]), float(v[2])
            if dx > 0 and abs(dy) < LANE_HALF_WIDTH and dx < self.front_dist:
                return False
        return True

    def _phi2_ttc(self, others: np.ndarray) -> bool:
        """φ₂: time-to-collision with any vehicle ahead in the current lane > ttc_sec."""
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy, dvx = float(v[1]), float(v[2]), float(v[3])
            # vehicle is ahead and in the same lane, and ego is approaching it
            if dx > 0 and abs(dy) < LANE_HALF_WIDTH and dvx < -1e-3:
                real_dx   = dx  * X_SCALE           # metres
                approach  = abs(dvx) * VX_SCALE     # m/s (relative)
                ttc       = real_dx / approach      # seconds
                if ttc < self.ttc_sec:
                    return False
        return True

    def _phi3_left_clear(self, others: np.ndarray) -> bool:
        """φ₃: left adjacent lane has no vehicle within side_margin longitudinally."""
        # Left lane has dy ≈ −LANE_WIDTH_NORM relative to ego
        target_dy = -LANE_WIDTH_NORM
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy = float(v[1]), float(v[2])
            if abs(dy - target_dy) < LANE_HALF_WIDTH * 1.5:
                if abs(dx) < self.side_margin:
                    return False
        return True

    def _phi4_right_clear(self, others: np.ndarray) -> bool:
        """φ₄: right adjacent lane has no vehicle within side_margin longitudinally."""
        target_dy = LANE_WIDTH_NORM
        for v in others:
            if v[0] < 0.5:
                continue
            dx, dy = float(v[1]), float(v[2])
            if abs(dy - target_dy) < LANE_HALF_WIDTH * 1.5:
                if abs(dx) < self.side_margin:
                    return False
        return True

    # ── public interface ──────────────────────────────────────────────────────

    def is_action_safe(self, obs: np.ndarray, action: int) -> bool:
        """
        Return True iff action satisfies the safety specification given obs.

        obs : shape (N_vehicles, N_features) — raw Kinematics observation
              (normalized, ego-relative; ego is row 0)
        """
        others = obs[1:]        # rows 1..N are other vehicles
        a = Action(action)

        if a == Action.LANE_LEFT:
            return self._phi3_left_clear(others)

        if a == Action.LANE_RIGHT:
            return self._phi4_right_clear(others)

        if a in (Action.IDLE, Action.FASTER):
            return self._phi1_front_distance(others) and self._phi2_ttc(others)

        # SLOWER only decelerates → never makes a forward collision worse
        return True

    def get_safe_action(self, obs: np.ndarray, proposed: int) -> int:
        """
        Return proposed if safe; otherwise substitute the highest-priority
        safe fallback: SLOWER > IDLE > LANE_LEFT > LANE_RIGHT.
        """
        self.total_steps += 1

        if self.is_action_safe(obs, proposed):
            return proposed

        self.override_steps += 1
        for fallback in (Action.SLOWER, Action.IDLE, Action.LANE_LEFT, Action.LANE_RIGHT):
            if self.is_action_safe(obs, int(fallback)):
                return int(fallback)

        return int(Action.SLOWER)   # guaranteed safe as final resort

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def override_rate(self) -> float:
        """Fraction of steps where the shield overrode the neural policy."""
        return self.override_steps / max(self.total_steps, 1)

    def reset_stats(self) -> None:
        self.total_steps    = 0
        self.override_steps = 0

    def __repr__(self) -> str:
        return (
            f"LTLSafetyShield(front_dist={self.front_dist}, "
            f"ttc_sec={self.ttc_sec}, side_margin={self.side_margin})"
        )
