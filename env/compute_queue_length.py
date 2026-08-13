"""
compute_queue_length.py

Computes mean queue length (halted-vehicle count per intersection,
sampled every 5 seconds) from the raw step-level CSVs already on
disk, for the medium-demand scalability comparison. No SUMO rerun
needed -- this data already exists in the step CSVs.
"""

import pandas as pd

GRIDS = ["2x2", "3x3", "4x4", "5x5"]
SEEDS = [11, 12, 13, 14, 15]
SCENARIO = "medium"

for grid in GRIDS:
    for controller in ["fixedtime", "dqn", "ppo"]:
        all_rows = []
        for seed in SEEDS:
            path = f"../results/grid{grid}/{controller}_{SCENARIO}_seed{seed}_steps.csv"
            df = pd.read_csv(path)
            all_rows.append(df)
        combined = pd.concat(all_rows, ignore_index=True)
        mean_queue = combined["queue_length"].mean()
        print(f"{grid} {controller}: mean queue length = {mean_queue:.2f}")
