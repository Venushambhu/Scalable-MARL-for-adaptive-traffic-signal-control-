"""
train_dqn_marl.py (Phase 2: generalized for any grid size)

TRUE independent multi-agent DQN training. One SB3 DQN model per
traffic-light intersection, auto-discovered from the network file
(no more hardcoded B1/B2/C1/C2). Each agent has its own replay
buffer and learns only from its own local transitions, sharing one
SUMO simulation.

Usage:
    python train_dqn_marl.py --grid 2x2 --scenario low --total_steps 20000
"""

import os
import sys
import csv
import argparse
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.utils import polyak_update

sys.path.append(".")
from traffic_env import TrafficSignalEnv, discover_tls_and_neighbours


class _DummySingleAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(6, dtype=np.float32), 0.0, True, False, {}


def linear_epsilon(step, total_steps, start=1.0, end=0.05, decay_fraction=0.7):
    decay_steps = total_steps * decay_fraction
    if step >= decay_steps:
        return end
    return start - (start - end) * (step / decay_steps)


def train(grid, scenario, total_steps=50000, seed=1):
    net_file = f"../network/grid{grid}/grid{grid}.net.xml"
    route_file = f"../routes/grid{grid}/{scenario}_demand.rou.xml"
    save_dir = f"../models/grid{grid}"
    log_dir = f"../results/grid{grid}/training_logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tls_ids, _ = discover_tls_and_neighbours(net_file)
    print(f"Grid {grid}: discovered {len(tls_ids)} agents: {tls_ids}")

    env = TrafficSignalEnv(net_file, route_file, num_seconds=1800,
                            decision_interval=5, seed=seed)

    models = {}
    for tls in tls_ids:
        models[tls] = DQN(
            "MlpPolicy",
            _DummySingleAgentEnv(),
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=500,
            batch_size=64,
            gamma=0.95,
            train_freq=1,
            target_update_interval=250,
            verbose=0,
            seed=seed,
        )
        models[tls].set_logger(configure_logger(folder=None, format_strings=[]))

    obs, info = env.reset()
    episode_reward = {tls: 0.0 for tls in tls_ids}
    episode_count = 0
    step = 0
    log_rows = []

    print(f"Starting DQN training: grid={grid}, scenario={scenario}, total_steps={total_steps}")

    while step < total_steps:
        epsilon = linear_epsilon(step, total_steps)
        actions = {}
        for tls in tls_ids:
            if random.random() < epsilon:
                actions[tls] = env.action_space.sample()
            else:
                action, _ = models[tls].predict(obs[tls], deterministic=True)
                actions[tls] = int(action)

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        for tls in tls_ids:
            models[tls].replay_buffer.add(
                obs[tls], next_obs[tls],
                np.array([actions[tls]]),
                np.array([rewards[tls]]),
                np.array([terminated or truncated]),
                [{}],
            )
            episode_reward[tls] += rewards[tls]

            if step > 500 and step % 4 == 0:
                models[tls].train(gradient_steps=1, batch_size=64)

            if step % models[tls].target_update_interval == 0:
                polyak_update(
                    models[tls].q_net.parameters(),
                    models[tls].q_net_target.parameters(),
                    models[tls].tau,
                )

        obs = next_obs
        step += 1

        if terminated or truncated:
            episode_count += 1
            avg_r = np.mean(list(episode_reward.values()))
            print(f"[step {step}] episode {episode_count} finished | "
                  f"avg reward/agent: {avg_r:.2f} | epsilon: {epsilon:.3f}")
            row = {"algorithm": "dqn", "grid": grid, "scenario": scenario,
                   "episode": episode_count, "step": step, "avg_reward": avg_r}
            for tls in tls_ids:
                row[f"reward_{tls}"] = episode_reward[tls]
            log_rows.append(row)
            episode_reward = {tls: 0.0 for tls in tls_ids}
            obs, info = env.reset()

    env.close()

    for tls in tls_ids:
        path = os.path.join(save_dir, f"dqn_{tls}_{scenario}.zip")
        models[tls].save(path)
        print(f"Saved {path}")

    log_path = os.path.join(log_dir, f"dqn_{scenario}_trainlog.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Saved training log to {log_path}")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="2x2", choices=["2x2", "3x3", "4x4", "5x5"])
    parser.add_argument("--scenario", required=True, choices=["low", "medium", "high", "dynamic"])
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    train(args.grid, args.scenario, args.total_steps, args.seed)
