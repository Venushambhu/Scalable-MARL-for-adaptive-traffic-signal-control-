"""
analyze_scalability.py (corrected per review)

Fixes from the friend's review:
- 95% CI now uses the Student-t critical value (df=n-1), not the z=1.96
  normal approximation, which is too narrow at n=5.
- Effect size relabeled as "Paired Cohen's dz" (mean of paired
  differences / SD of paired differences), not generic Cohen's d.
- Added completion rate (%) using known loaded-vehicle counts per grid,
  since raw throughput isn't comparable across grids with different
  calibrated demand levels.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

GRIDS = ["2x2", "3x3", "4x4", "5x5"]
NUM_AGENTS = {"2x2": 4, "3x3": 9, "4x4": 16, "5x5": 25}
LOADED_VEHICLES = {"2x2": 1500, "3x3": 1800, "4x4": 2000, "5x5": 1566}
SCENARIO = "medium"
SEEDS = [11, 12, 13, 14, 15]
N = len(SEEDS)
T_CRIT = stats.t.ppf(0.975, df=N-1)  # ~2.776 at df=4, correct for small-n CI

OUT_DIR = "../results/scalability_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def paired_cohens_dz(a, b):
    diff = a - b
    return diff.mean() / diff.std(ddof=1)


def load_master():
    rows = []
    for grid in GRIDS:
        result_dir = f"../results/grid{grid}"
        for controller in ["fixedtime", "dqn", "ppo"]:
            for seed in SEEDS:
                path = os.path.join(result_dir, f"{controller}_{SCENARIO}_seed{seed}_summary.csv")
                if not os.path.exists(path):
                    print(f"WARNING: missing {path}")
                    continue
                df = pd.read_csv(path)
                df["num_agents"] = NUM_AGENTS[grid]
                df["loaded_vehicles"] = LOADED_VEHICLES[grid]
                df["completion_rate_pct"] = 100 * df["completed_trips"] / LOADED_VEHICLES[grid]
                rows.append(df)
    master = pd.concat(rows, ignore_index=True)
    master.to_csv(os.path.join(OUT_DIR, "master_results.csv"), index=False)
    print(f"Master results: {len(master)} rows saved to master_results.csv")
    return master


def scalability_table(master):
    metrics = ["avg_travel_time", "avg_waiting_time", "completed_trips", "completion_rate_pct"]
    results = []

    for grid in GRIDS:
        n_agents = NUM_AGENTS[grid]
        sub = master[master["num_agents"] == n_agents]
        fixed = sub[sub["controller"] == "fixedtime"].sort_values("seed").reset_index(drop=True)

        for controller in ["dqn", "ppo"]:
            other = sub[sub["controller"] == controller].sort_values("seed").reset_index(drop=True)
            for metric in metrics:
                if len(other) != len(fixed) or len(fixed) == 0:
                    continue
                mean_o, std_o = other[metric].mean(), other[metric].std()
                ci_halfwidth = T_CRIT * std_o / np.sqrt(len(other))
                pct = 100 * (mean_o - fixed[metric].mean()) / fixed[metric].mean()
                t_stat, p = stats.ttest_rel(other[metric], fixed[metric])
                dz = paired_cohens_dz(other[metric], fixed[metric])

                results.append({
                    "grid": grid, "num_agents": n_agents, "controller": controller, "metric": metric,
                    "mean": round(mean_o, 2), "std": round(std_o, 2),
                    "ci95_lower": round(mean_o - ci_halfwidth, 2), "ci95_upper": round(mean_o + ci_halfwidth, 2),
                    "pct_change_vs_fixed": round(pct, 1),
                    "t_stat": round(t_stat, 3), "p_value": round(p, 4),
                    "paired_cohens_dz": round(dz, 3),
                    "significant_p05": p < 0.05,
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "scalability_summary.csv"), index=False)
    print(f"\nUsing t-critical (df={N-1}) = {T_CRIT:.3f} for 95% CI (was incorrectly using z=1.96)")
    print("\n=== Scalability summary (corrected CI + paired Cohen's dz) ===")
    print(results_df.to_string(index=False))
    return results_df


def make_scalability_plots(master):
    metrics = ["avg_travel_time", "avg_waiting_time", "completion_rate_pct"]
    titles = {
        "avg_travel_time": "Average Travel Time (s)",
        "avg_waiting_time": "Average Waiting Time (s)",
        "completion_rate_pct": "Completion Rate (%)",
    }
    colors = {"fixedtime": "#888888", "dqn": "#4C72B0", "ppo": "#DD8452"}
    x_vals = [NUM_AGENTS[g] for g in GRIDS]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for controller in ["fixedtime", "dqn", "ppo"]:
            means, stds = [], []
            for grid in GRIDS:
                sub = master[(master["num_agents"] == NUM_AGENTS[grid]) & (master["controller"] == controller)]
                means.append(sub[metric].mean())
                stds.append(sub[metric].std())
            ax.errorbar(x_vals, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                        label=controller.upper() if controller != "fixedtime" else "Fixed-Time",
                        color=colors[controller])

        ax.set_xlabel("Number of Controlled Intersections")
        ax.set_ylabel(titles[metric])
        ax.set_xticks(x_vals)
        ax.set_title(f"{titles[metric]} vs Network Size (Medium Demand)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"scalability_{metric}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for controller in ["dqn", "ppo"]:
        pct_improvements = []
        for grid in GRIDS:
            n_agents = NUM_AGENTS[grid]
            sub = master[master["num_agents"] == n_agents]
            fixed_mean = sub[sub["controller"] == "fixedtime"]["avg_waiting_time"].mean()
            other_mean = sub[sub["controller"] == controller]["avg_waiting_time"].mean()
            pct_improvements.append(100 * (other_mean - fixed_mean) / fixed_mean)
        ax.plot(x_vals, pct_improvements, marker='o', linewidth=2,
                label=controller.upper(), color=colors[controller])
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Number of Controlled Intersections")
    ax.set_ylabel("% Change in Avg Waiting Time vs Fixed-Time")
    ax.set_xticks(x_vals)
    ax.set_title("MARL Improvement Over Fixed-Time Control vs Network Size")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "scalability_improvement.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    master = load_master()
    scalability_table(master)
    make_scalability_plots(master)
    print("\nScalability analysis complete (corrected). See ../results/scalability_analysis/")
