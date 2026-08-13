"""
analyze_results.py

Consolidates all *_summary.csv files (fixedtime / dqn / ppo x 4 scenarios
x 5 seeds), computes descriptive statistics, runs paired significance
tests (paired t-test + Wilcoxon signed-rank, since every controller ran
on the SAME 5 seeds per scenario), and generates the summary table and
comparison plots for Chapter 4.

Usage:
    python analyze_results.py
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

RESULTS_DIR = "../results/grid2x2"
OUT_DIR = "../results/grid2x2/analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def load_all_summaries():
    files = glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["seed"].between(11, 15)].reset_index(drop=True)
    return df


def descriptive_table(df):
    metrics = ["avg_travel_time", "avg_waiting_time", "completed_trips"]
    desc = df.groupby(["scenario", "controller"])[metrics].agg(["mean", "std"])
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))
    print("\n=== Descriptive statistics (mean, std across 5 seeds) ===")
    print(desc.round(2))
    return desc


def paired_tests(df):
    metrics = ["avg_travel_time", "avg_waiting_time", "completed_trips"]
    scenarios = df["scenario"].unique()
    results = []

    for scenario in scenarios:
        sub = df[df["scenario"] == scenario]
        fixed = sub[sub["controller"] == "fixedtime"].sort_values("seed").reset_index(drop=True)
        dqn = sub[sub["controller"] == "dqn"].sort_values("seed").reset_index(drop=True)
        ppo = sub[sub["controller"] == "ppo"].sort_values("seed").reset_index(drop=True)

        for metric in metrics:
            for name, other in [("DQN_vs_Fixed", dqn), ("PPO_vs_Fixed", ppo)]:
                if len(other) == len(fixed) and len(fixed) > 1:
                    t_stat, t_p = stats.ttest_rel(other[metric], fixed[metric])
                    try:
                        w_stat, w_p = stats.wilcoxon(other[metric], fixed[metric])
                    except ValueError:
                        w_stat, w_p = np.nan, np.nan

                    pct_change = 100 * (other[metric].mean() - fixed[metric].mean()) / fixed[metric].mean()

                    results.append({
                        "scenario": scenario,
                        "metric": metric,
                        "comparison": name,
                        "mean_fixed": round(fixed[metric].mean(), 2),
                        "mean_other": round(other[metric].mean(), 2),
                        "pct_change": round(pct_change, 1),
                        "t_stat": round(t_stat, 3),
                        "t_pvalue": round(t_p, 4),
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_pvalue": round(w_p, 4) if not np.isnan(w_p) else np.nan,
                        "significant_p05": t_p < 0.05,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "significance_tests.csv"), index=False)
    print("\n=== Paired significance tests (DQN/PPO vs Fixed-Time) ===")
    print(results_df.to_string(index=False))
    return results_df


def make_plots(df):
    metrics = ["avg_travel_time", "avg_waiting_time", "completed_trips"]
    titles = {
        "avg_travel_time": "Average Travel Time (s)",
        "avg_waiting_time": "Average Waiting Time (s)",
        "completed_trips": "Throughput (completed trips)",
    }
    scenarios = ["low", "medium", "high", "dynamic"]
    controllers = ["fixedtime", "dqn", "ppo"]
    colors = {"fixedtime": "#888888", "dqn": "#4C72B0", "ppo": "#DD8452"}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(scenarios))
        width = 0.25

        for i, controller in enumerate(controllers):
            means, stds = [], []
            for scenario in scenarios:
                sub = df[(df["scenario"] == scenario) & (df["controller"] == controller)][metric]
                means.append(sub.mean())
                stds.append(sub.std())
            ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                   label=controller.upper() if controller != "fixedtime" else "Fixed-Time",
                   color=colors[controller])

        ax.set_xticks(x + width)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_ylabel(titles[metric])
        ax.set_title(f"{titles[metric]} by Controller and Demand Scenario")
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"plot_{metric}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    df = load_all_summaries()
    print(f"Loaded {len(df)} summary rows across all controllers/scenarios/seeds.")
    descriptive_table(df)
    paired_tests(df)
    make_plots(df)
    print("\nAnalysis complete. See ../results/analysis/ for tables and plots.")
