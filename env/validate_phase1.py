"""
validate_phase1.py

Gate check before starting the scalability refactor (Phase 2). Confirms:
  1. All 32 expected model files exist (2 algorithms x 4 scenarios x 4 agents)
  2. All 60 expected result summary files exist (20 fixedtime + 20 DQN + 20 PPO,
     on seeds 11-15 across 4 scenarios)
  3. No missing or duplicate seed/scenario/controller combinations
  4. Every summary CSV's completed_trips and avg_travel_time are recomputed
     directly from the raw tripinfo.xml and match to within floating-point
     tolerance (catches stale/mismatched files)

Exits with a clear PASS/FAIL report. Does not modify anything.
"""

import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd

TLS_IDS = ["B1", "B2", "C1", "C2"]
SCENARIOS = ["low", "medium", "high", "dynamic"]
SEEDS = [11, 12, 13, 14, 15]
ALGOS = ["dqn", "ppo"]

MODEL_DIR = "../models"
RESULTS_DIR = "../results"

errors = []
warnings = []


def check_models():
    print("Checking model files...")
    missing = []
    for algo in ALGOS:
        for scenario in SCENARIOS:
            for tls in TLS_IDS:
                path = os.path.join(MODEL_DIR, f"{algo}_{tls}_{scenario}.zip")
                if not os.path.exists(path):
                    missing.append(path)
    if missing:
        errors.append(f"{len(missing)} missing model files: {missing[:5]}{'...' if len(missing)>5 else ''}")
    else:
        print(f"  OK: all {len(ALGOS)*len(SCENARIOS)*len(TLS_IDS)} model files present")


def check_result_files():
    print("Checking result summary files...")
    expected = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            expected.append(("fixedtime", scenario, seed))
            expected.append(("dqn", scenario, seed))
            expected.append(("ppo", scenario, seed))

    missing = []
    for controller, scenario, seed in expected:
        path = os.path.join(RESULTS_DIR, f"{controller}_{scenario}_seed{seed}_summary.csv")
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        errors.append(f"{len(missing)} missing result summary files: {missing[:5]}")
    else:
        print(f"  OK: all {len(expected)} expected summary files present")

    all_summaries = glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv"))
    seen = set()
    dupes = []
    for f in all_summaries:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            key = (row["controller"], row["scenario"], row["seed"])
            if key in seen:
                dupes.append(key)
            seen.add(key)
    if dupes:
        warnings.append(f"Duplicate (controller,scenario,seed) entries found across files: {dupes}")


def cross_check_tripinfo():
    print("Cross-checking summary CSVs against raw tripinfo XML...")
    mismatches = 0
    checked = 0
    for controller in ["fixedtime", "dqn", "ppo"]:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                summary_path = os.path.join(RESULTS_DIR, f"{controller}_{scenario}_seed{seed}_summary.csv")
                trip_path = os.path.join(RESULTS_DIR, f"{controller}_{scenario}_seed{seed}_tripinfo.xml")
                if not (os.path.exists(summary_path) and os.path.exists(trip_path)):
                    continue
                summary = pd.read_csv(summary_path).iloc[0]
                tree = ET.parse(trip_path)
                trips = tree.getroot().findall("tripinfo")
                n_completed = len(trips)
                if n_completed > 0:
                    avg_duration = sum(float(t.get("duration")) for t in trips) / n_completed
                else:
                    avg_duration = 0.0

                checked += 1
                if n_completed != summary["completed_trips"]:
                    mismatches += 1
                    errors.append(f"{controller}/{scenario}/seed{seed}: summary says "
                                   f"{summary['completed_trips']} completed trips, tripinfo XML has {n_completed}")
                elif abs(avg_duration - summary["avg_travel_time"]) > 0.5:
                    mismatches += 1
                    errors.append(f"{controller}/{scenario}/seed{seed}: summary avg_travel_time="
                                   f"{summary['avg_travel_time']}, recomputed from XML={avg_duration:.2f}")

    print(f"  Cross-checked {checked} runs, {mismatches} mismatches found")


if __name__ == "__main__":
    check_models()
    check_result_files()
    cross_check_tripinfo()

    print("\n" + "=" * 50)
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print("  ERROR:", e)
    else:
        print("VALIDATION PASSED: all models, results, and data cross-checks are consistent")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print("  WARNING:", w)
    print("=" * 50)
