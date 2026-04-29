#!/usr/bin/env python3
"""
Generate comparison figures from saved results.

Figures produced (saved to results/figures/)
--------------------------------------------
  fig1_training_curves.svg   Episode reward vs training timestep
                             (neural vs neurosymbolic)
  fig2_bar_comparison.svg    Bar charts: reward / speed / crash rate
                             for all three agents
  fig3_shield_override.svg   Shield override rate over training
                             (neurosymbolic only)

Usage
-----
  python plot_results.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR = "results/figures"
EARLY_CRASH_STEPS = 5  # episodes crashed at or before this step are treated as spawn-induced
COLORS = {
    "neural": "#E69F00",        # Okabe-Ito orange
    "symbolic": "#56B4E9",      # Okabe-Ito sky blue
    "neurosymbolic": "#009E73", # Okabe-Ito bluish-green
}
LABELS = {
    "neural": "Pure Neural (Pearl DQN)",
    "symbolic": "Pure Symbolic (FSM)",
    "neurosymbolic": "NeuroSymbolic (Shielded Pearl DQN)",
}


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ── Fig 1: training curves ────────────────────────────────────────────────────


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for agent in ("neural", "neurosymbolic"):
        data = load_json(f"results/{agent}_train_curve.json")
        if data is None:
            continue
        curve = data["curve"]
        xs = [p["timestep"] for p in curve]
        means = [p["mean_reward"] for p in curve]
        stds = [p["std_reward"] for p in curve]
        crashes = [p["crash_rate"] for p in curve]
        col = COLORS[agent]
        label = LABELS[agent]

        ax = axes[0]
        ax.plot(xs, means, color=col, label=label, linewidth=2)
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=col,
            alpha=0.2,
        )

        ax = axes[1]
        ax.plot(xs, crashes, color=col, label=label, linewidth=2)

    for ax, title, ylabel in zip(
        axes,
        ["Reward during training", "Crash rate during training"],
        ["Mean episode reward", "Crash rate"],
    ):
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Training Progress: Neural vs NeuroSymbolic", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_training_curves.svg")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── Fig 2: bar comparison ─────────────────────────────────────────────────────


def plot_bar_comparison() -> None:
    summary = load_json("results/summary.json")
    if summary is None:
        return

    agents = [a for a in ("neural", "symbolic", "neurosymbolic") if a in summary]
    if not agents:
        return

    metrics = [
        ("mean_reward", "std_reward", "Mean episode reward", None),
        ("mean_speed", "std_speed", "Mean speed (m/s)", None),
        ("crash_rate", None, "Crash rate", "%"),
    ]
    x = np.arange(len(agents))
    width = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (metric, std_key, title, fmt) in zip(axes, metrics):
        vals = [summary[a][metric] for a in agents]
        errs = [summary[a][std_key] for a in agents] if std_key else None
        cols = [COLORS[a] for a in agents]
        bars = ax.bar(
            x,
            vals,
            width,
            yerr=errs,
            color=cols,
            capsize=4,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[a] for a in agents], fontsize=8, rotation=15, ha="right"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if fmt == "%":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        # annotate bar tops
        for bar, val in zip(bars, vals):
            label = f"{val:.1%}" if fmt == "%" else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) * 0.01),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Agent Comparison: Reward / Speed / Safety", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_bar_comparison.svg")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── Fig 3: shield filter rate during training ────────────────────────────────


def plot_shield_override() -> None:
    ns_data = load_json("results/neurosymbolic_train_curve.json")
    neural_data = load_json("results/neural_train_curve.json")
    if ns_data is None:
        return

    curve = ns_data.get("curve", [])
    if not curve:
        print("[INFO] Neurosymbolic training curve is empty — skipping Fig 3")
        return

    xs = [p["timestep"] for p in curve]
    ns_crashes = [p["crash_rate"] for p in curve]
    neural_crashes = (
        [p["crash_rate"] for p in neural_data.get("curve", [])]
        if neural_data
        else []
    )
    # Final shield filter rate stored at the top level of the JSON
    filter_rate = ns_data.get("final_shield_filter_rate")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Crash rate curves
    ax.plot(xs, ns_crashes, color=COLORS["neurosymbolic"], linewidth=2,
            label=LABELS["neurosymbolic"])
    if neural_crashes and len(neural_crashes) == len(xs):
        ax.plot(xs, neural_crashes, color=COLORS["neural"], linewidth=2,
                linestyle="--", label=LABELS["neural"])

    # Annotate final shield filter rate as a horizontal reference line
    if filter_rate is not None:
        ax.axhline(
            filter_rate,
            color=COLORS["neurosymbolic"],
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            label=f"Shield filter rate (final): {filter_rate:.1%}",
        )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Rate")
    ax.set_title(
        "Crash Rate vs Training Progress\n"
        "(shield filter rate shown as dotted reference line)"
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_shield_filter.svg")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── episode-level distribution ────────────────────────────────────────────────


def plot_reward_distribution() -> None:
    comparison = load_json("results/comparison.json")
    if comparison is None:
        return

    agents = [a for a in ("neural", "symbolic", "neurosymbolic") if a in comparison]
    fig, ax = plt.subplots(figsize=(8, 4))

    for agent in agents:
        rewards = [ep["reward"] for ep in comparison[agent]]
        ax.hist(
            rewards,
            bins=15,
            alpha=0.55,
            color=COLORS[agent],
            label=LABELS[agent],
            edgecolor="white",
        )

    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution per Episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_reward_distribution.svg")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── Fig 5: bar comparison excluding early-crash episodes ─────────────────────


def plot_bar_comparison_filtered() -> None:
    comparison = load_json("results/comparison.json")
    if comparison is None:
        return

    agents = [a for a in ("neural", "symbolic", "neurosymbolic") if a in comparison]
    if not agents:
        return

    def _stats(eps: list) -> dict | None:
        kept = [e for e in eps if not (e["crashed"] and e["steps"] <= EARLY_CRASH_STEPS)]
        n = len(kept)
        if n == 0:
            return None
        rewards = [e["reward"] for e in kept]
        speeds = [e["mean_speed"] for e in kept]
        crashes = [e["crashed"] for e in kept]
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_speed": float(np.mean(speeds)),
            "std_speed": float(np.std(speeds)),
            "crash_rate": float(np.mean(crashes)),
            "n": n,
        }

    stats = {a: _stats(comparison[a]) for a in agents}
    agents = [a for a in agents if stats[a] is not None]
    if not agents:
        return

    ns_label = ", ".join(f"{a}: n={stats[a]['n']}" for a in agents)
    metrics = [
        ("mean_reward", "std_reward", "Mean episode reward", None),
        ("mean_speed", "std_speed", "Mean speed (m/s)", None),
        ("crash_rate", None, "Crash rate", "%"),
    ]
    x = np.arange(len(agents))
    width = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (metric, std_key, title, fmt) in zip(axes, metrics):
        vals = [stats[a][metric] for a in agents]
        errs = [stats[a][std_key] for a in agents] if std_key else None
        cols = [COLORS[a] for a in agents]
        bars = ax.bar(
            x,
            vals,
            width,
            yerr=errs,
            color=cols,
            capsize=4,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[a] for a in agents], fontsize=8, rotation=15, ha="right"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if fmt == "%":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, vals):
            label = f"{val:.1%}" if fmt == "%" else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) * 0.01),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        f"Agent Comparison (early crashes ≤{EARLY_CRASH_STEPS} steps excluded)\n"
        f"Reward / Speed / Safety  [{ns_label}]",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_bar_comparison_filtered.svg")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_training_curves()
    plot_bar_comparison()
    plot_shield_override()
    plot_reward_distribution()
    plot_bar_comparison_filtered()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
