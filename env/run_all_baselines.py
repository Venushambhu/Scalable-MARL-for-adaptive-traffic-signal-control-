from collect_baseline import run_baseline

SCENARIOS = ["low", "medium", "high", "dynamic"]
SEEDS = [11, 12, 13, 14, 15]

if __name__ == "__main__":
    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(f"\n=== Running fixed-time baseline: {scenario}, seed {seed} ===")
            run_baseline(scenario, seed)