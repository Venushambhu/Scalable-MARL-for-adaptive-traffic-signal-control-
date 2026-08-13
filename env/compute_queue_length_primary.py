import pandas as pd

SCENARIOS = ["low", "medium", "high", "dynamic"]
SEEDS = [11, 12, 13, 14, 15]

for scenario in SCENARIOS:
    for controller in ["fixedtime", "dqn", "ppo"]:
        all_rows = []
        for seed in SEEDS:
            path = f"../results/grid2x2/{controller}_{scenario}_seed{seed}_steps.csv"
            df = pd.read_csv(path)
            all_rows.append(df)
        combined = pd.concat(all_rows, ignore_index=True)
        mean_queue = combined["queue_length"].mean()
        std_queue = combined.groupby("seed")["queue_length"].mean().std()
        print(f"{scenario} {controller}: mean queue length = {mean_queue:.2f} (seed-level std = {std_queue:.2f})")
