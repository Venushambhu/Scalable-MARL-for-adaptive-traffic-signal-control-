"""
plot_learning_curves.py

Builds learning-curve figures from the training logs. One 2x2 grid
figure, one subplot per demand scenario (each with its OWN y-axis
scale, since reward magnitude differs hugely by demand level), with
DQN and PPO overlaid in each subplot for direct comparison.

Usage:
    python plot_learning_curves.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "../results/training_logs"
OUT_DIR = "../results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = ["low", "medium", "high", "dynamic"]
ALGO_COLORS = {"dqn": "#4C72B0", "ppo": "#DD8452"}
ROLL_WINDOW = 10


def plot_grid():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, scenario in zip(axes, SCENARIOS):
        for algo in ["dqn", "ppo"]:
            path = os.path.join(LOG_DIR, f"{algo}_{scenario}_trainlog.csv")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            df = pd.read_csv(path)
            smoothed = df["avg_reward"].rolling(ROLL_WINDOW, min_periods=1).mean()
            ax.plot(df["episode"], df["avg_reward"], color=ALGO_COLORS[algo], alpha=0.18, linewidth=1)
            ax.plot(df["episode"], smoothed, color=ALGO_COLORS[algo], linewidth=2.2,
                     label=algo.upper())

        ax.set_title(f"{scenario.capitalize()} Demand", fontsize=11, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Reward per Agent")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(f"DQN vs PPO Learning Curves by Demand Scenario\n"
                  f"(faint = raw per-episode reward, bold = {ROLL_WINDOW}-episode rolling mean)",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "learning_curves_by_scenario.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_grid()
    print("Done.")
