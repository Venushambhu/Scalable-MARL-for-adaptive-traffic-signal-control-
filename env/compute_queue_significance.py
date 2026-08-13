import pandas as pd
from scipy import stats

SCENARIOS = ["low", "medium", "high", "dynamic"]
SEEDS = [11, 12, 13, 14, 15]

def per_seed_means(scenario, controller):
    means = []
    for seed in SEEDS:
        path = f"../results/grid2x2/{controller}_{scenario}_seed{seed}_steps.csv"
        df = pd.read_csv(path)
        means.append(df["queue_length"].mean())
    return pd.Series(means)

for scenario in SCENARIOS:
    fixed = per_seed_means(scenario, "fixedtime")
    for controller in ["dqn", "ppo"]:
        other = per_seed_means(scenario, controller)
        t_stat, p = stats.ttest_rel(other, fixed)
        pct = 100 * (other.mean() - fixed.mean()) / fixed.mean()
        print(f"{scenario} {controller}_vs_fixed: mean_fixed={fixed.mean():.2f}, mean_other={other.mean():.2f}, "
              f"pct={pct:+.1f}%, t={t_stat:.2f}, p={p:.4f}, sig={p<0.05}")
