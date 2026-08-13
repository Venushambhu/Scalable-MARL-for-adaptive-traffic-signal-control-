from evaluate_trained import run_evaluation

CONTROLLERS = ["dqn", "ppo"]
SCENARIOS = ["low", "medium", "high", "dynamic"]
SEEDS = [11, 12, 13, 14, 15]

if __name__ == "__main__":
    for controller in CONTROLLERS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                print(f"\n=== Evaluating {controller} on {scenario}, seed {seed} ===")
                run_evaluation(controller, scenario, seed)