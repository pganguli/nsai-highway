#!/usr/bin/env python3
"""
Train all learnable agents using Pearl DQN and save results.

Usage
-----
  python train.py               # train both neural and neurosymbolic
  python train.py --agent neural
  python train.py --agent neurosymbolic
  python train.py --timesteps 50000
  python train.py --resume      # continue from latest checkpoint

Outputs (written to ./models/ and ./results/)
-------
  models/neural/model.pth                          final agent state dict
  models/neural/best_model.pth                     best checkpoint (by eval reward)
  models/neural/checkpoints/pearl_model_N_steps.pth
  results/neural_train_curve.json                  written incrementally
  results/neurosymbolic_train_curve.json

Architecture change from SB3
-----------------------------
The shielded agent previously used a post-hoc action override (SB3
ShieldedEnv wrapper) which introduced an off-policy error: the replay
buffer stored (obs, proposed_action, reward, next_obs) but the *executed*
action may have differed.

The Pearl implementation uses action masking at the source:
  1. ShieldedHighwayPearlEnv returns only safe actions as available_action_space.
  2. LTLShieldSafetyModule filters actions before each act() call.
  3. The Bellman backup uses the stored safe next-action set, so targets are
     computed over only the feasible actions in the next state.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401

from pearl.pearl_agent import PearlAgent
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from config import (
    BASE_ENV_CONFIG,
    N_OBS_VEHICLES,
    N_FEATURES,
    TOTAL_TIMESTEPS,
    EVAL_FREQ,
    DQN_KWARGS,
)
from deep_set_network import DeepSetQNetwork
from double_dqn_per import DoubleDQNWithPER
from per_replay_buffer import PrioritizedReplayBuffer
from pearl_environment import make_pearl_env, make_pearl_eval_env, N_ACTIONS
from pearl_safety_module import LTLShieldSafetyModule
from safety_shield import LTLSafetyShield

STATE_DIM = N_OBS_VEHICLES * N_FEATURES  # 50


def _make_agent(
    total_timesteps: int, shielded: bool = False
) -> tuple[PearlAgent, LTLSafetyShield | None]:
    """Construct a Pearl Double-DQN+PER+DeepSet agent."""
    exploration_module = EGreedyExploration(
        epsilon=0.05,
        start_epsilon=1.0,
        end_epsilon=0.05,
        warmup_steps=int(total_timesteps * 0.1),
    )

    network = DeepSetQNetwork(
        n_vehicles=N_OBS_VEHICLES,
        n_features=N_FEATURES,
        action_dim=N_ACTIONS,  # one-hot action representation
        phi_hidden=[64, 64],  # per-vehicle encoder
        rho_hidden=[256, 256],  # Q-value head
    )

    policy_learner = DoubleDQNWithPER(
        state_dim=STATE_DIM,
        action_space=DiscreteActionSpace([torch.tensor([i]) for i in range(N_ACTIONS)]),
        network_instance=network,
        learning_rate=DQN_KWARGS["learning_rate"],
        discount_factor=DQN_KWARGS["gamma"],
        training_rounds=DQN_KWARGS["gradient_steps"],
        batch_size=DQN_KWARGS["batch_size"],
        target_update_freq=DQN_KWARGS["target_update_interval"],
        soft_update_tau=1.0,  # hard target update
        exploration_module=exploration_module,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=N_ACTIONS,
        ),
    )

    shield = LTLSafetyShield() if shielded else None
    safety_module = LTLShieldSafetyModule(shield=shield) if shielded else None

    return PearlAgent(
        policy_learner=policy_learner,
        replay_buffer=PrioritizedReplayBuffer(
            DQN_KWARGS["buffer_size"],
            alpha=0.6,
            beta=0.4,
            beta_annealing_steps=total_timesteps,
        ),
        safety_module=safety_module,
    ), shield


def _latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return path of the highest-step Pearl checkpoint in ckpt_dir, or None."""
    files = glob.glob(os.path.join(ckpt_dir, "pearl_model_*_steps.pth"))
    if not files:
        return None

    def _step(p: str) -> int:
        return int(os.path.basename(p).split("_")[2])

    return max(files, key=_step)


# ── evaluation helper ─────────────────────────────────────────────────────────


def _eval_agent(
    agent: PearlAgent,
    eval_env,
    n_episodes: int,
) -> tuple[list[float], float, float]:
    """
    Run n_episodes greedy evaluation episodes.

    Uses the policy_learner and safety_module directly (no replay buffer push)
    so training state is not perturbed.

    Returns (rewards, crash_rate, mean_speed).
    """
    pl = agent.policy_learner
    sm = agent.safety_module
    device = agent.device

    rewards: list[float] = []
    crashes: list[float] = []
    speeds: list[float] = []

    for _ in range(n_episodes):
        obs, action_space = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_crashed = False
        ep_speeds: list[float] = []

        while not done:
            obs_t = torch.as_tensor(obs).to(device)
            action_space.to(device)
            safe_space = sm.filter_action(obs_t, action_space)
            safe_space.to(device)  # pyright: ignore[reportAttributeAccessIssue]
            action = pl.act(
                subjective_state=obs_t, available_action_space=safe_space, exploit=True
            )

            action_result = eval_env.step(action)
            ep_reward += float(action_result.reward)
            info = action_result.info or {}
            ep_crashed = ep_crashed or bool(info.get("crashed", False))
            ep_speeds.append(float(info.get("speed", 0.0)))
            obs = action_result.observation
            action_space = action_result.available_action_space or eval_env.action_space
            done = action_result.done

        rewards.append(ep_reward)
        crashes.append(float(ep_crashed))
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)

    return rewards, float(np.mean(crashes)), float(np.mean(speeds))


# ── core training loop ────────────────────────────────────────────────────────


def _train(
    label: str,
    model_dir: str,
    curve_path: str,
    shielded: bool,
    timesteps: int,
    seed: int,
    resume: bool,
    ckpt_freq: int,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build environments
    shield = LTLSafetyShield() if shielded else None
    env = (
        make_pearl_env(BASE_ENV_CONFIG, shielded=True, fast=True)
        if shielded
        else make_pearl_env(BASE_ENV_CONFIG, shielded=False, fast=True)
    )
    eval_env = make_pearl_eval_env(shielded=shielded)

    # Share the same shield instance between env wrapper and safety module
    if shielded and shield is not None:
        env.shield = shield  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(env, "shield"):
            # Replace the env's auto-created shield with our named one
            env.shield = shield  # pyright: ignore[reportAttributeAccessIssue]

    agent, _shield = _make_agent(timesteps, shielded=shielded)  # pyright: ignore[reportGeneralTypeIssues]

    # Resume from checkpoint
    start_step = 0
    curve: list[dict] = []
    best_reward = float("-inf")

    if resume:
        latest_ckpt = _latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            print(f"  Resuming from: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, weights_only=False)
            agent.load_state_dict(ckpt["agent_state"])
            start_step = ckpt["timestep"]
            best_reward = ckpt.get("best_reward", float("-inf"))
            print(f"  Continuing from step {start_step}")

    if resume and os.path.exists(curve_path):
        with open(curve_path) as f:
            saved = json.load(f).get("curve", [])
            curve = saved
            if saved:
                best_reward = max(best_reward, max(p["mean_reward"] for p in saved))

    # Initialise env and agent state
    obs, action_space = env.reset()
    agent.reset(obs, action_space)

    learning_starts = DQN_KWARGS["learning_starts"]
    t0 = time.time()

    with tqdm(total=timesteps, initial=0, desc=label) as pbar:
        for step_in_run in range(timesteps):
            t = start_step + step_in_run

            action = agent.act(exploit=False)
            action_result = env.step(action)
            agent.observe(action_result)

            if step_in_run >= learning_starts:
                agent.learn()

            if action_result.done:
                obs, action_space = env.reset()
                agent.reset(obs, action_space)

            pbar.update(1)

            # Periodic checkpoint
            if ckpt_freq > 0 and (step_in_run + 1) % ckpt_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, f"pearl_model_{t + 1}_steps.pth")
                torch.save(
                    {
                        "agent_state": agent.state_dict(),
                        "timestep": t + 1,
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )

            # Periodic evaluation
            if (step_in_run + 1) % EVAL_FREQ == 0:
                ep_rewards, crash_r, mean_sp = _eval_agent(
                    agent, eval_env, n_episodes=5
                )
                mean_r = float(np.mean(ep_rewards))
                std_r = float(np.std(ep_rewards))

                is_best = mean_r > best_reward
                if is_best:
                    best_reward = mean_r
                    torch.save(
                        {
                            "agent_state": agent.state_dict(),
                            "timestep": t + 1,
                            "best_reward": best_reward,
                        },
                        os.path.join(model_dir, "best_model.pth"),
                    )

                curve.append(
                    {
                        "timestep": t + 1,
                        "mean_reward": mean_r,
                        "std_reward": std_r,
                        "crash_rate": crash_r,
                        "mean_speed": mean_sp,
                    }
                )
                print(
                    f"\n  [{t + 1:>7d}] "
                    f"reward={mean_r:.3f}±{std_r:.2f}  "
                    f"crash={crash_r:.2f}  "
                    f"speed={mean_sp:.1f} m/s" + ("  ★ new best" if is_best else "")
                )
                with open(curve_path, "w") as f:
                    json.dump({"curve": curve}, f, indent=2)

    elapsed = time.time() - t0

    # Save final model
    torch.save(
        {
            "agent_state": agent.state_dict(),
            "timestep": start_step + timesteps,
            "best_reward": best_reward,
        },
        os.path.join(model_dir, "model.pth"),
    )

    with open(curve_path) as f:
        data = json.load(f)
    data["training_seconds"] = elapsed
    if shielded and hasattr(env, "filter_rate"):
        data["final_shield_filter_rate"] = env.filter_rate  # pyright: ignore[reportAttributeAccessIssue]
        print(f"  Shield filter rate during training: {env.filter_rate:.2%}")  # pyright: ignore[reportAttributeAccessIssue]
    with open(curve_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{label} saved → {model_dir}/model.pth  ({elapsed:.0f}s)")
    env.close()
    eval_env.close()


# ── public entry points ───────────────────────────────────────────────────────


def train_neural(timesteps: int, seed: int, resume: bool, ckpt_freq: int) -> None:
    print("\n=== Training PURE NEURAL (Pearl DQN) ===")
    _train(
        label="Neural DQN",
        model_dir="models/neural",
        curve_path="results/neural_train_curve.json",
        shielded=False,
        timesteps=timesteps,
        seed=seed,
        resume=resume,
        ckpt_freq=ckpt_freq,
    )


def train_neurosymbolic(
    timesteps: int, seed: int, resume: bool, ckpt_freq: int
) -> None:
    print("\n=== Training NEUROSYMBOLIC (Pearl Shielded DQN) ===")
    _train(
        label="Shielded DQN",
        model_dir="models/neurosymbolic",
        curve_path="results/neurosymbolic_train_curve.json",
        shielded=True,
        timesteps=timesteps,
        seed=seed,
        resume=resume,
        ckpt_freq=ckpt_freq,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train highway agents with Pearl DQN")
    parser.add_argument(
        "--agent", choices=["neural", "neurosymbolic", "all"], default="all"
    )
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true", help="continue from the latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=EVAL_FREQ,
        help="save a checkpoint every N timesteps (default: EVAL_FREQ)",
    )
    args = parser.parse_args()

    if args.agent in ("neural", "all"):
        train_neural(args.timesteps, args.seed, args.resume, args.checkpoint_freq)

    if args.agent in ("neurosymbolic", "all"):
        train_neurosymbolic(
            args.timesteps, args.seed, args.resume, args.checkpoint_freq
        )

    print("\nTraining complete.  Run  python evaluate.py  to compare all agents.")


if __name__ == "__main__":
    main()
