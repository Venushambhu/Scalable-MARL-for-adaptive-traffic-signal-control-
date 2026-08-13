from train_ppo_marl import train

SCENARIOS = ["low", "medium", "high", "dynamic"]

if __name__ == "__main__":
    for scenario in SCENARIOS:
        print(f"\n\n=========== Training PPO on scenario: {scenario} ===========")
        train(scenario, total_steps=50000, seed=1)