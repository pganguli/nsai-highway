"""
Shared configuration for all three agents (neural / symbolic / neurosymbolic).

Observation: Kinematics, 10 vehicles x 5 features, normalized, ego-relative.
  features: [presence, x, y, vx, vy]
  x  - longitudinal distance from ego (+forward)   range [-100, 100] m
  y  - lateral distance from ego     (+right)       range [-100, 100] m
  vx - longitudinal relative speed                  range [-30,   30] m/s
  vy - lateral      relative speed                  range [-30,   30] m/s

Action space: DiscreteMetaAction
  0 LANE_LEFT   1 IDLE   2 LANE_RIGHT   3 FASTER   4 SLOWER
"""

# ── observation ──────────────────────────────────────────────────────────────
N_OBS_VEHICLES = 10
N_FEATURES = 5

OBS_CONFIG = {
    "type": "Kinematics",
    "vehicles_count": N_OBS_VEHICLES,
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-30, 30],
        "vy": [-30, 30],
    },
    "normalize": True,
    "absolute": False,
    "order": "sorted",
    "see_behind": True,   # expose vehicles behind ego so the shield can detect
                          # fast-approaching cut-ins and rear occupants of target lanes
}

# ── environment ───────────────────────────────────────────────────────────────
BASE_ENV_CONFIG = {
    "observation": OBS_CONFIG,
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,  # seconds per episode
    "initial_spacing": 2,
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}

# Longer episodes for evaluation to get stable statistics
EVAL_ENV_CONFIG = {**BASE_ENV_CONFIG, "duration": 60}

# ── DQN hyperparameters ───────────────────────────────────────────────────────
DQN_KWARGS = {
    "policy": "MlpPolicy",
    "policy_kwargs": {"net_arch": [256, 256]},
    "learning_rate": 5e-4,
    "buffer_size": 15_000,
    "learning_starts": 200,
    "batch_size": 32,
    "gamma": 0.8,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 50,
    "verbose": 1,
}

TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 20
EVAL_FREQ = 3_000  # record eval metrics every N training steps

# ── safety-shield thresholds (normalized observation space) ───────────────────
# lane width ≈ 4 m → 4/100 = 0.04 normalized
# "same lane"      |dy| < LANE_HALF_WIDTH
# "adjacent lane"  |dy - ±LANE_WIDTH| < LANE_HALF_WIDTH

X_SCALE = 100.0  # m
VX_SCALE = 30.0  # m/s
LANE_WIDTH_NORM = 0.04   # 4 m
# Use 0.75× lane-width as the "same-lane" half-width (3 m either side of
# centre).  The physical collision threshold is 0.5× (2 m), but widening to
# 0.75× catches vehicles mid-lane-change before they fully enter ego's lane.
LANE_HALF_WIDTH = LANE_WIDTH_NORM * 0.75  # 0.03

SHIELD_FRONT_DIST = 0.20  # 20 m minimum following distance
SHIELD_TTC_SEC = 2.0  # 2 s minimum time-to-collision
SHIELD_SIDE_MARGIN = 0.20  # 20 m longitudinal clearance for lane change

# Reward penalty added whenever the shield overrides the neural policy's action.
# This gives the DQN an explicit signal to stop proposing actions the shield
# has to correct, so it gradually learns to internalise the safety constraints.
# With normalize_reward=True, per-step rewards are O(0.1), so 0.05 is noticeable
# but not large enough to swamp the speed/lane reward.
SHIELD_OVERRIDE_PENALTY = 0.05
