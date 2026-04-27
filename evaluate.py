"""
Evaluate and compare all three agents.

Usage
-----
  python evaluate.py                    # evaluate all, use default model paths
  python evaluate.py --episodes 30
  python evaluate.py --render           # show environment window

Outputs
-------
  results/comparison.json    per-episode metrics for each agent
  results/summary.json       mean ± std summary table (printed to stdout too)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import highway_env  # noqa: F401

from config import N_EVAL_EPISODES
from pearl_environment import make_pearl_eval_env
from pearl_safety_module import LTLShieldSafetyModule
from symbolic_agent import SymbolicAgent
from environments import make_eval_env  # plain gym env for symbolic agent
from train_all import _make_agent


def load_pearl_agent(model_path: str, shielded: bool = False) -> object:
    """Load a Pearl agent from a saved .pth checkpoint."""
    agent, _shield = _make_agent(total_timesteps=100_000, shielded=shielded)
    ckpt = torch.load(model_path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state"])
    return agent


def run_pearl_episode(env, agent, render: bool = False) -> dict:
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


def run_symbolic_episode(env, agent: SymbolicAgent, render: bool = False) -> dict:
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
    agent, env, n_episodes: int, label: str, render: bool = False
) -> list[dict]:
    print(f"\n── {label} ({n_episodes} episodes) ──")
    results = []
    for i in range(n_episodes):
        ep = run_pearl_episode(env, agent, render=render)
        results.append(ep)
        print(
            f"  ep {i + 1:>3d}  reward={ep['reward']:+.3f}  "
            f"speed={ep['mean_speed']:.1f} m/s  "
            f"crash={'YES' if ep['crashed'] else ' no'}"
        )
    return results


def evaluate_symbolic(
    agent, env, n_episodes: int, label: str, render: bool = False
) -> list[dict]:
    print(f"\n── {label} ({n_episodes} episodes) ──")
    results = []
    for i in range(n_episodes):
        ep = run_symbolic_episode(env, agent, render=render)
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
    parser.add_argument("--neural-model", default="models/neural/model.pth")
    parser.add_argument(
        "--neurosymbolic-model", default="models/neurosymbolic/model.pth"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    # ── 1. Pure Neural (Pearl DQN) ────────────────────────────────────────────
    if os.path.exists(args.neural_model):
        env = make_pearl_eval_env(shielded=False)
        agent = load_pearl_agent(args.neural_model, shielded=False)
        all_results["neural"] = evaluate_pearl(
            agent, env, args.episodes, "Pure Neural (Pearl DQN)", args.render
        )
        env.close()
    else:
        print(f"[SKIP] Neural model not found at {args.neural_model}")
        print("       Run  python train_all.py --agent neural  first.")

    # ── 2. Pure Symbolic ─────────────────────────────────────────────────────
    sym_env = make_eval_env(shielded=False)
    all_results["symbolic"] = evaluate_symbolic(
        SymbolicAgent(), sym_env, args.episodes, "Pure Symbolic (FSM)", args.render
    )
    sym_env.close()

    # ── 3. NeuroSymbolic (Shielded Pearl DQN) ────────────────────────────────
    if os.path.exists(args.neurosymbolic_model):
        env = make_pearl_eval_env(shielded=True)
        agent = load_pearl_agent(args.neurosymbolic_model, shielded=True)
        all_results["neurosymbolic"] = evaluate_pearl(
            agent, env, args.episodes, "NeuroSymbolic (Shielded Pearl DQN)", args.render
        )
        if hasattr(env, "filter_rate"):
            print(f"  Shield filter rate (eval): {env.filter_rate:.2%}")
        env.close()
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
    print("Run    python plot_results.py  to generate figures.")


if __name__ == "__main__":
    main()
