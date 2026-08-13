"""
validate_and_summarize_grid.py

Grid-aware version of the Phase 1 validation gate + a descriptive
summary, for any single grid/scenario combination. Confirms all
15 runs exist, cross-checks against raw tripinfo XML, then prints
mean/std per controller and paired t-tests vs fixed-time.

Usage:
    python validate_and_summarize_grid.py --grid 3x3 --scenario medium
"""

import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from scipy import stats

SEEDS = [11, 12, 13, 14, 15]
CONTROLLERS = ["fixedtime", "dqn", "ppo"]


def validate(grid, scenario):
    out_dir = f"../results/grid{grid}"
    errors = []
    rows = []

    for controller in CONTROLLERS:
        for seed in SEEDS:
            summary_path = os.path.join(out_dir, f"{controller}_{scenario}_seed{seed}_summary.csv")
            trip_path = os.path.join(out_dir, f"{controller}_{scenario}_seed{seed}_tripinfo.xml")
            if not os.path.exists(summary_path):
                errors.append(f"MISSING: {summary_path}")
                continue
            if not os.path.exists(trip_path):
                errors.append(f"MISSING: {trip_path}")
                continue

            summary = pd.read_csv(summary_path).iloc[0]
            tree = ET.parse(trip_path)
            trips = tree.getroot().findall("tripinfo")
            n_completed = len(trips)
            if n_completed != summary["completed_trips"]:
                errors.append(f"MISMATCH {controller}/seed{seed}: summary={summary['completed_trips']}, xml={n_completed}")

            rows.append(summary)

    print(f"Validated {len(rows)}/{len(CONTROLLERS)*len(SEEDS)} runs for grid={grid}, scenario={scenario}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print("  ", e)
        return None
    else:
        print("VALIDATION PASSED")
        return pd.DataFrame(rows)


def summarize(df, grid, scenario):
    print(f"\n=== Descriptive stats: grid={grid}, scenario={scenario} ===")
    metrics = ["avg_travel_time", "avg_waiting_time", "completed_trips"]
    desc = df.groupby("controller")[metrics].agg(["mean", "std"])
    print(desc.round(2))

    print(f"\n=== Paired t-tests vs Fixed-Time ===")
    fixed = df[df["controller"] == "fixedtime"].sort_values("seed").reset_index(drop=True)
    for controller in ["dqn", "ppo"]:
        other = df[df["controller"] == controller].sort_values("seed").reset_index(drop=True)
        for metric in metrics:
            t_stat, p = stats.ttest_rel(other[metric], fixed[metric])
            pct = 100 * (other[metric].mean() - fixed[metric].mean()) / fixed[metric].mean()
            sig = "YES" if p < 0.05 else "no"
            print(f"  {controller.upper()} {metric}: {pct:+.1f}%, t={t_stat:.2f}, p={p:.4f}, significant={sig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    df = validate(args.grid, args.scenario)
    if df is not None:
        summarize(df, args.grid, args.scenario)
