"""
Evaluate and compare all three agents.

Usage
-----
  python evaluate.py                    # evaluate all, use default model paths
  python evaluate.py --episodes 30
  python evaluate.py --render           # show environment window
  python evaluate.py --video            # record MP4s to results/videos/

Outputs
-------
  results/comparison.json              per-episode metrics for each agent
  results/summary.json                 mean ± std summary table (printed to stdout too)
  results/videos/{agent}/              MP4 per episode (with --video)
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.dirname(__file__))

import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401

from config import EVAL_ENV_CONFIG, N_EVAL_EPISODES
from pearl_environment import (
    HighwayPearlEnv,
    ShieldedHighwayPearlEnv,
    make_pearl_eval_env,
)
from symbolic_agent import SymbolicAgent
from environments import make_eval_env  # plain gym env for symbolic agent
from train_all import _make_agent


def _make_pearl_video_env(shielded: bool, video_dir: str):
    """Create a Pearl eval env that records every episode to video_dir."""
    os.makedirs(video_dir, exist_ok=True)
    gym_env = gym.make("highway-v0", config=EVAL_ENV_CONFIG, render_mode="rgb_array")
    recorder = RecordVideo(
        gym_env,
        video_folder=video_dir,
        episode_trigger=lambda _ep: True,
        name_prefix="ep",
        disable_logger=True,
    )
    # Let highway-env push intermediate simulation frames into the recorder.
    recorder.unwrapped.set_record_video_wrapper(recorder)  # pyright: ignore[reportAttributeAccessIssue]
    if shielded:
        return ShieldedHighwayPearlEnv(recorder)
    return HighwayPearlEnv(recorder)


def _make_symbolic_video_env(video_dir: str) -> gym.Env:
    """Create a plain gym eval env that records every episode to video_dir."""
    os.makedirs(video_dir, exist_ok=True)
    gym_env = gym.make("highway-v0", config=EVAL_ENV_CONFIG, render_mode="rgb_array")
    recorder = RecordVideo(
        gym_env,
        video_folder=video_dir,
        episode_trigger=lambda _ep: True,
        name_prefix="ep",
        disable_logger=True,
    )
    recorder.unwrapped.set_record_video_wrapper(recorder)  # pyright: ignore[reportAttributeAccessIssue]
    return recorder


def load_pearl_agent(model_path: str, shielded: bool = False) -> object:
    """Load a Pearl agent from a saved .pth checkpoint."""
    agent, _shield = _make_agent(total_timesteps=100_000, shielded=shielded)  # pyright: ignore[reportGeneralTypeIssues]
    ckpt = torch.load(model_path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state"])
    return agent


def run_pearl_episode(
    env, agent, render: bool = False, render_video: bool = False
) -> dict:
    """Run one greedy episode with a Pearl agent (no replay-buffer writes)."""
    pl = agent.policy_learner
    sm = agent.safety_module
    device = agent.device

    obs, action_space = env.reset()
    done = False
    ep_reward = 0.0
    ep_speeds: list[float] = []
    ep_crashed = False

    while not done:
        if render:
            try:
                env._env.render()
            except Exception:
                pass

        obs_t = torch.as_tensor(obs).to(device)
        action_space.to(device)
        safe_space = sm.filter_action(obs_t, action_space)
        safe_space.to(device)
        action = pl.act(
            subjective_state=obs_t, available_action_space=safe_space, exploit=True
        )

        action_result = env.step(action)
        if render_video:
            env._env.render()  # flush intermediate simulation frames into recorder
        ep_reward += float(action_result.reward)
        info = action_result.info or {}
        ep_speeds.append(float(info.get("speed", 0.0)))
        ep_crashed = ep_crashed or bool(info.get("crashed", False))
        obs = action_result.observation
        action_space = action_result.available_action_space or env.action_space
        done = action_result.done

    return {
        "reward": ep_reward,
        "mean_speed": float(np.mean(ep_speeds)) if ep_speeds else 0.0,
        "crashed": int(ep_crashed),
        "steps": len(ep_speeds),
    }


def run_symbolic_episode(
    env, agent: SymbolicAgent, render: bool = False, render_video: bool = False
) -> dict:
    """Run one episode with the rule-based symbolic agent."""
    obs, _ = env.reset()
    done = truncated = False
    ep_reward = 0.0
    ep_speeds: list[float] = []
    ep_crashed = False

    while not (done or truncated):
        if render:
            env.render()
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        if render_video:
            env.render()  # flush intermediate simulation frames into recorder
        ep_reward += float(reward)
        ep_speeds.append(float(info.get("speed", 0.0)))
        ep_crashed = ep_crashed or bool(info.get("crashed", False))

    return {
        "reward": ep_reward,
        "mean_speed": float(np.mean(ep_speeds)) if ep_speeds else 0.0,
        "crashed": int(ep_crashed),
        "steps": len(ep_speeds),
    }


def evaluate_pearl(
    agent,
    env,
    n_episodes: int,
    label: str,
    render: bool = False,
    render_video: bool = False,
) -> list[dict]:
    print(f"\n── {label} ({n_episodes} episodes) ──")
    results = []
    for i in range(n_episodes):
        ep = run_pearl_episode(env, agent, render=render, render_video=render_video)
        results.append(ep)
        print(
            f"  ep {i + 1:>3d}  reward={ep['reward']:+.3f}  "
            f"speed={ep['mean_speed']:.1f} m/s  "
            f"crash={'YES' if ep['crashed'] else ' no'}"
        )
    return results


def evaluate_symbolic(
    agent,
    env,
    n_episodes: int,
    label: str,
    render: bool = False,
    render_video: bool = False,
) -> list[dict]:
    print(f"\n── {label} ({n_episodes} episodes) ──")
    results = []
    for i in range(n_episodes):
        ep = run_symbolic_episode(env, agent, render=render, render_video=render_video)
        results.append(ep)
        print(
            f"  ep {i + 1:>3d}  reward={ep['reward']:+.3f}  "
            f"speed={ep['mean_speed']:.1f} m/s  "
            f"crash={'YES' if ep['crashed'] else ' no'}"
        )
    return results


def summarise(results: list[dict]) -> dict:
    rewards = [r["reward"] for r in results]
    speeds = [r["mean_speed"] for r in results]
    crashes = [r["crashed"] for r in results]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_speed": float(np.mean(speeds)),
        "std_speed": float(np.std(speeds)),
        "crash_rate": float(np.mean(crashes)),
        "n_episodes": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare agents")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--video",
        action="store_true",
        help="record MP4 for every episode into results/videos/",
    )
    parser.add_argument("--neural-model", default="models/neural/model.pth")
    parser.add_argument(
        "--neurosymbolic-model", default="models/neurosymbolic/model.pth"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    # ── 1. Pure Neural (Pearl DQN) ────────────────────────────────────────────
    if os.path.exists(args.neural_model):
        if args.video:
            env = _make_pearl_video_env(
                shielded=False, video_dir="results/videos/neural"
            )
        else:
            env = make_pearl_eval_env(shielded=False)
        agent = load_pearl_agent(args.neural_model, shielded=False)
        all_results["neural"] = evaluate_pearl(
            agent,
            env,
            args.episodes,
            "Pure Neural (Pearl DQN)",
            args.render,
            render_video=args.video,
        )
        env.close()
        if args.video:
            print("  Videos → results/videos/neural/")
    else:
        print(f"[SKIP] Neural model not found at {args.neural_model}")
        print("       Run  python train_all.py --agent neural  first.")

    # ── 2. Pure Symbolic ─────────────────────────────────────────────────────
    if args.video:
        sym_env = _make_symbolic_video_env(video_dir="results/videos/symbolic")
    else:
        sym_env = make_eval_env(shielded=False)
    all_results["symbolic"] = evaluate_symbolic(
        SymbolicAgent(),
        sym_env,
        args.episodes,
        "Pure Symbolic (FSM)",
        args.render,
        render_video=args.video,
    )
    sym_env.close()
    if args.video:
        print("  Videos → results/videos/symbolic/")

    # ── 3. NeuroSymbolic (Shielded Pearl DQN) ────────────────────────────────
    if os.path.exists(args.neurosymbolic_model):
        if args.video:
            env = _make_pearl_video_env(
                shielded=True, video_dir="results/videos/neurosymbolic"
            )
        else:
            env = make_pearl_eval_env(shielded=True)
        agent = load_pearl_agent(args.neurosymbolic_model, shielded=True)
        all_results["neurosymbolic"] = evaluate_pearl(
            agent,
            env,
            args.episodes,
            "NeuroSymbolic (Shielded Pearl DQN)",
            args.render,
            render_video=args.video,
        )
        if hasattr(env, "filter_rate"):
            print(f"  Shield filter rate (eval): {env.filter_rate:.2%}")  # pyright: ignore[reportAttributeAccessIssue]
        env.close()
        if args.video:
            print("  Videos → results/videos/neurosymbolic/")
    else:
        print(f"[SKIP] Neurosymbolic model not found at {args.neurosymbolic_model}")
        print("       Run  python train_all.py --agent neurosymbolic  first.")

    # ── summary table ─────────────────────────────────────────────────────────
    summaries = {k: summarise(v) for k, v in all_results.items()}

    print("\n" + "=" * 65)
    print(f"{'Agent':<20} {'Reward':>10} {'Speed':>10} {'Crash%':>8}")
    print("-" * 65)
    for name, s in summaries.items():
        print(
            f"{name:<20} "
            f"{s['mean_reward']:>+8.3f}±{s['std_reward']:.2f}  "
            f"{s['mean_speed']:>7.1f} m/s  "
            f"{s['crash_rate']:>6.1%}"
        )
    print("=" * 65)

    with open("results/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("results/summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nSaved → results/comparison.json  results/summary.json")
    if args.video:
        print("Videos → results/videos/{neural,symbolic,neurosymbolic}/")
    print("Run    python plot_results.py  to generate figures.")


if __name__ == "__main__":
    main()
