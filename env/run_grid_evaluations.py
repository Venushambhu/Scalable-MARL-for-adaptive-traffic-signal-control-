"""
run_grid_evaluations.py

Runs fixed-time baseline + DQN + PPO evaluation for a given grid size
and scenario, across seeds 11-15 (15 runs total per grid/scenario).

Usage:
    python run_grid_evaluations.py --grid 3x3 --scenario medium
"""

import argparse
from collect_baseline import run_baseline
from evaluate_trained import run_evaluation

SEEDS = [11, 12, 13, 14, 15]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, choices=["2x2", "3x3", "4x4", "5x5"])
    parser.add_argument("--scenario", required=True, choices=["low", "medium", "high", "dynamic"])
    args = parser.parse_args()

    for seed in SEEDS:
        print(f"\n=== Fixed-time: grid={args.grid}, scenario={args.scenario}, seed={seed} ===")
        run_baseline(args.grid, args.scenario, seed)

    for controller in ["dqn", "ppo"]:
        for seed in SEEDS:
            print(f"\n=== {controller.upper()}: grid={args.grid}, scenario={args.scenario}, seed={seed} ===")
            run_evaluation(args.grid, controller, args.scenario, seed)

    print("\nAll 15 evaluation runs complete.")
