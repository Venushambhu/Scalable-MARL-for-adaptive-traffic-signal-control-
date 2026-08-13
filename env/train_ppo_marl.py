"""
train_ppo_marl.py (Phase 2: generalized for any grid size)

TRUE independent multi-agent PPO training. One SB3 PPO model per
traffic-light intersection, auto-discovered from the network file.
Each agent has its own rollout buffer, manually filled and trained
(bypassing SB3's .learn(), since it doesn't support multi-agent
training natively).

Usage:
    python train_ppo_marl.py --grid 2x2 --scenario low --total_steps 50000
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as configure_logger

sys.path.append(".")
from traffic_env import TrafficSignalEnv, discover_tls_and_neighbours

N_STEPS = 128


class _DummySingleAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(6, dtype=np.float32), 0.0, True, False, {}


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
        models[tls] = PPO(
            "MlpPolicy",
            _DummySingleAgentEnv(),
            n_steps=N_STEPS,
            batch_size=32,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
        )
        models[tls].set_logger(configure_logger(folder=None, format_strings=[]))
        models[tls].rollout_buffer.reset()

    obs, info = env.reset()
    episode_reward = {tls: 0.0 for tls in tls_ids}
    episode_start = {tls: np.array([True]) for tls in tls_ids}
    episode_count = 0
    step = 0
    buffer_pos = 0
    log_rows = []

    print(f"Starting PPO training: grid={grid}, scenario={scenario}, total_steps={total_steps}")

    while step < total_steps:
        actions, values, log_probs = {}, {}, {}

        for tls in tls_ids:
            obs_tensor, _ = models[tls].policy.obs_to_tensor(obs[tls])
            with th.no_grad():
                act, val, logp = models[tls].policy(obs_tensor)
            actions[tls] = int(act.item())
            values[tls] = val
            log_probs[tls] = logp

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        for tls in tls_ids:
            models[tls].rollout_buffer.add(
                obs[tls],
                np.array([actions[tls]]),
                np.array([rewards[tls]]),
                episode_start[tls],
                values[tls],
                log_probs[tls],
            )
            episode_reward[tls] += rewards[tls]
            episode_start[tls] = np.array([done])

        obs = next_obs
        step += 1
        buffer_pos += 1

        if buffer_pos >= N_STEPS:
            for tls in tls_ids:
                obs_tensor, _ = models[tls].policy.obs_to_tensor(obs[tls])
                with th.no_grad():
                    last_values = models[tls].policy.predict_values(obs_tensor)
                models[tls].rollout_buffer.compute_returns_and_advantage(
                    last_values=last_values, dones=np.array([done])
                )
                models[tls].train()
                models[tls].rollout_buffer.reset()
            buffer_pos = 0

        if done:
            episode_count += 1
            avg_r = np.mean(list(episode_reward.values()))
            print(f"[step {step}] episode {episode_count} finished | avg reward/agent: {avg_r:.2f}")
            row = {"algorithm": "ppo", "grid": grid, "scenario": scenario,
                   "episode": episode_count, "step": step, "avg_reward": avg_r}
            for tls in tls_ids:
                row[f"reward_{tls}"] = episode_reward[tls]
            log_rows.append(row)
            episode_reward = {tls: 0.0 for tls in tls_ids}
            obs, info = env.reset()
            for tls in tls_ids:
                episode_start[tls] = np.array([True])

    env.close()

    for tls in tls_ids:
        path = os.path.join(save_dir, f"ppo_{tls}_{scenario}.zip")
        models[tls].save(path)
        print(f"Saved {path}")

    log_path = os.path.join(log_dir, f"ppo_{scenario}_trainlog.csv")
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
