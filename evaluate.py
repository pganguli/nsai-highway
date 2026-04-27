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

sys.path.insert(0, os.path.dirname(__file__))

import highway_env  # noqa: F401
from stable_baselines3 import DQN

from config import N_EVAL_EPISODES
from environments import make_eval_env, ShieldedEnv
from symbolic_agent import SymbolicAgent


# ── episode runner ────────────────────────────────────────────────────────────


def run_episode(env, agent, render: bool = False) -> dict:
    """
    Run one episode and return a metrics dict.

    agent must expose  predict(obs, deterministic=True) → (action, state).
    """
    obs, _ = env.reset()
    done = truncated = False
    ep_reward = 0.0
    ep_speeds = []
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


def evaluate_agent(
    agent,
    env,
    n_episodes: int,
    label: str,
    render: bool = False,
) -> list[dict]:
    print(f"\n── {label} ({n_episodes} episodes) ──")
    results = []
    for i in range(n_episodes):
        ep = run_episode(env, agent, render=render)
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare agents")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--neural-model", default="models/neural/best_model")
    parser.add_argument("--neurosymbolic-model", default="models/neurosymbolic/best_model")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    # ── 1. Pure Neural ────────────────────────────────────────────────────────
    if os.path.exists(args.neural_model + ".zip"):
        env = make_eval_env(shielded=False)
        model = DQN.load(args.neural_model, env=env)
        all_results["neural"] = evaluate_agent(
            model, env, args.episodes, "Pure Neural (DQN)", args.render
        )
        env.close()
    else:
        print(f"[SKIP] Neural model not found at {args.neural_model}.zip")
        print("       Run  python train_all.py --agent neural  first.")

    # ── 2. Pure Symbolic ─────────────────────────────────────────────────────
    env = make_eval_env(shielded=False)
    all_results["symbolic"] = evaluate_agent(
        SymbolicAgent(), env, args.episodes, "Pure Symbolic (FSM)", args.render
    )
    env.close()

    # ── 3. NeuroSymbolic (Shielded DQN) ──────────────────────────────────────
    if os.path.exists(args.neurosymbolic_model + ".zip"):
        env = make_eval_env(shielded=True)
        model = DQN.load(args.neurosymbolic_model, env=env)
        all_results["neurosymbolic"] = evaluate_agent(
            model, env, args.episodes, "NeuroSymbolic (Shielded DQN)", args.render
        )

        if isinstance(env, ShieldedEnv):
            print(f"  Shield override rate (eval): {env.shield.override_rate:.2%}")
        env.close()
    else:
        print(f"[SKIP] Neurosymbolic model not found at {args.neurosymbolic_model}.zip")
        print("       Run  python train_all.py --agent neurosymbolic  first.")

    # ── summary table ─────────────────────────────────────────────────────────
    summaries: dict[str, dict] = {k: summarise(v) for k, v in all_results.items()}

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

    # ── save ──────────────────────────────────────────────────────────────────
    with open("results/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("results/summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nSaved → results/comparison.json  results/summary.json")
    print("Run    python plot_results.py  to generate figures.")


if __name__ == "__main__":
    main()
