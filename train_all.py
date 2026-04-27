"""
Train all learnable agents and save results.

Usage
-----
  python train_all.py               # train both neural and neurosymbolic
  python train_all.py --agent neural
  python train_all.py --agent neurosymbolic
  python train_all.py --timesteps 50000
  python train_all.py --resume      # continue from latest checkpoint

Outputs (written to ./models/ and ./results/)
-------
  models/neural/model.zip                        final model
  models/neural/checkpoints/rl_model_*_steps.zip periodic checkpoints
  models/neurosymbolic/model.zip
  models/neurosymbolic/checkpoints/
  results/neural_train_curve.json                written incrementally
  results/neurosymbolic_train_curve.json
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))

import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from config import (
    BASE_ENV_CONFIG,
    DQN_KWARGS,
    TOTAL_TIMESTEPS,
    EVAL_FREQ,
)
from environments import make_env, make_eval_env


def _latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the path of the highest-step checkpoint in ckpt_dir, or None."""
    pattern = os.path.join(ckpt_dir, "rl_model_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    # filenames: rl_model_<step>_steps.zip
    def _step(p: str) -> int:
        name = os.path.basename(p)          # rl_model_12000_steps.zip
        return int(name.split("_")[2])
    return max(files, key=_step)


# ── callback to record training progress ─────────────────────────────────────


class TrainingCurveCallback(BaseCallback):
    """
    Records mean episode reward (and collision rate) at regular intervals
    by running a short evaluation rollout against a separate eval env.
    Writes the curve to disk after every eval so progress survives interruption.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        curve_path: str,
        best_model_path: str,
        n_eval_ep: int = 5,
    ):
        super().__init__()
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.curve_path      = curve_path
        self.best_model_path = best_model_path
        self.n_eval_ep       = n_eval_ep
        self.curve: list[dict] = []
        self._best_reward    = float("-inf")
        # load any curve written by a previous run so resume appends cleanly
        if os.path.exists(curve_path):
            with open(curve_path) as f:
                saved = json.load(f).get("curve", [])
                self.curve = saved
                if saved:
                    self._best_reward = max(p["mean_reward"] for p in saved)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            rewards, crash_rates, speeds = [], [], []
            for _ in range(self.n_eval_ep):
                ep_reward, ep_crashed, ep_speed = 0.0, False, []
                obs, _ = self.eval_env.reset()
                done = truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, truncated, info = self.eval_env.step(action)
                    ep_reward += r  # type: ignore
                    ep_crashed = ep_crashed or info.get("crashed", False)
                    ep_speed.append(info.get("speed", 0.0))
                rewards.append(ep_reward)
                crash_rates.append(float(ep_crashed))
                speeds.append(float(np.mean(ep_speed)) if ep_speed else 0.0)

            mean_reward = float(np.mean(rewards))
            self.curve.append(
                {
                    "timestep":    self.num_timesteps,
                    "mean_reward": mean_reward,
                    "std_reward":  float(np.std(rewards)),
                    "crash_rate":  float(np.mean(crash_rates)),
                    "mean_speed":  float(np.mean(speeds)),
                }
            )
            is_best = mean_reward > self._best_reward
            if is_best:
                self._best_reward = mean_reward
                self.model.save(self.best_model_path)
            print(
                f"  [{self.num_timesteps:>7d}] "
                f"reward={mean_reward:.3f}  "
                f"crash={self.curve[-1]['crash_rate']:.2f}  "
                f"speed={self.curve[-1]['mean_speed']:.1f} m/s"
                + ("  ★ new best" if is_best else "")
            )
            # flush to disk so the curve survives an interruption
            with open(self.curve_path, "w") as f:
                json.dump({"curve": self.curve}, f, indent=2)
        return True


# ── training helpers ──────────────────────────────────────────────────────────


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
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env      = make_env(BASE_ENV_CONFIG, shielded=shielded)
    eval_env = make_eval_env(shielded=shielded)

    ckpt_dir           = os.path.join(model_dir, "checkpoints")
    latest_ckpt        = _latest_checkpoint(ckpt_dir)
    reset_num_timesteps = True

    if resume and latest_ckpt:
        print(f"  Resuming from checkpoint: {latest_ckpt}")
        model = DQN.load(latest_ckpt, env=env)
        reset_num_timesteps = False
    else:
        tb_log = f"results/tensorboard/{os.path.basename(model_dir)}/"
        model  = DQN(env=env, **DQN_KWARGS, seed=seed, tensorboard_log=tb_log)

    curve_cb = TrainingCurveCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        curve_path=curve_path,
        best_model_path=os.path.join(model_dir, "best_model"),
        n_eval_ep=5,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=ckpt_dir,
        name_prefix="rl_model",
        verbose=0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([curve_cb, ckpt_cb]),
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    model.save(os.path.join(model_dir, "model"))

    # write final training seconds into the curve file (curve entries already flushed)
    with open(curve_path) as f:
        data = json.load(f)
    data["training_seconds"] = elapsed
    if shielded and hasattr(env, "shield"):
        rate = env.shield.override_rate  # type: ignore
        data["final_shield_override_rate"] = rate
        print(f"  Shield override rate during training: {rate:.2%}")
    with open(curve_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{label} saved → {model_dir}/model.zip  ({elapsed:.0f}s)")
    env.close()
    eval_env.close()


def train_neural(timesteps: int, seed: int, resume: bool, ckpt_freq: int) -> None:
    print("\n=== Training PURE NEURAL (DQN) ===")
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


def train_neurosymbolic(timesteps: int, seed: int, resume: bool, ckpt_freq: int) -> None:
    print("\n=== Training NEUROSYMBOLIC (Shielded DQN) ===")
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


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train highway agents")
    parser.add_argument("--agent", choices=["neural", "neurosymbolic", "all"], default="all")
    parser.add_argument("--timesteps",    type=int,  default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--resume",       action="store_true",
                        help="continue from the latest checkpoint in models/*/checkpoints/")
    parser.add_argument("--checkpoint-freq", type=int, default=EVAL_FREQ,
                        help="save a checkpoint every N timesteps (default: EVAL_FREQ)")
    args = parser.parse_args()

    if args.agent in ("neural", "all"):
        train_neural(args.timesteps, args.seed, args.resume, args.checkpoint_freq)

    if args.agent in ("neurosymbolic", "all"):
        train_neurosymbolic(args.timesteps, args.seed, args.resume, args.checkpoint_freq)

    print("\nTraining complete.  Run  python evaluate.py  to compare all agents.")


if __name__ == "__main__":
    main()
