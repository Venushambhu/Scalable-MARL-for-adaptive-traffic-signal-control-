"""
train_ppo_marl_v2.py

Independent multi-agent PPO training for Thesis Revision V2.

One Stable-Baselines3 PPO model per traffic-light intersection.
Agents share one SUMO simulation but maintain independent policies
and independent rollout buffers.

Revision V2:
- TrafficSignalEnvV2 corrected signal-control semantics
- original thesis PPO hyperparameters preserved
- deterministic seeding
- runtime measurement
- peak process-tree RAM
- rollout-buffer allocation measurement
- saved-model size measurement
- isolated models_v2/results_v2 outputs
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import psutil
import torch as th

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as configure_logger

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    MODELS_V2_DIR,
    RESULTS_V2_DIR,
    GRID_AGENT_COUNTS,
)

from traffic_env_v2 import (
    TrafficSignalEnv,
    discover_tls_and_neighbours,
)


# ============================================================
# ORIGINAL PPO HYPERPARAMETERS
# ============================================================

N_STEPS = 128
BATCH_SIZE = 32
N_EPOCHS = 4

LEARNING_RATE = 3e-4
GAMMA = 0.95
GAE_LAMBDA = 0.90
CLIP_RANGE = 0.20
ENT_COEF = 0.01

SIM_DURATION = 1800
DECISION_INTERVAL = 5

MEMORY_SAMPLE_INTERVAL = 50


# ============================================================
# DUMMY SINGLE-AGENT ENV
# ============================================================

class DummySingleAgentEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        return (
            np.zeros(
                6,
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action):

        return (
            np.zeros(
                6,
                dtype=np.float32,
            ),
            0.0,
            True,
            False,
            {},
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


# ============================================================
# PROCESS RAM
# ============================================================

def process_tree_rss_mb():

    root = psutil.Process(
        os.getpid()
    )

    processes = [root]

    try:

        processes.extend(
            root.children(
                recursive=True
            )
        )

    except psutil.Error:

        pass

    total = 0

    for process in processes:

        try:

            total += (
                process.memory_info().rss
            )

        except psutil.Error:

            continue

    return (
        total
        / 1024.0
        / 1024.0
    )


# ============================================================
# ROLLOUT BUFFER MEMORY
# ============================================================

def rollout_buffer_bytes(buffer):

    attributes = [
        "observations",
        "actions",
        "rewards",
        "returns",
        "episode_starts",
        "values",
        "log_probs",
        "advantages",
    ]

    total = 0
    seen = set()

    for name in attributes:

        value = getattr(
            buffer,
            name,
            None,
        )

        if not isinstance(
            value,
            np.ndarray,
        ):
            continue

        identity = id(value)

        if identity in seen:
            continue

        seen.add(identity)

        total += value.nbytes

    return total


# ============================================================
# MODEL SIZE
# ============================================================

def total_model_size_mb(paths):

    return (
        sum(
            Path(path).stat().st_size
            for path in paths
            if Path(path).exists()
        )
        / 1024.0
        / 1024.0
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    grid,
    scenario,
    total_steps=50_000,
    seed=1,
):

    key = (
        grid,
        scenario,
    )

    if key not in ROUTE_FILES:

        raise KeyError(
            f"No route configured for "
            f"{grid}/{scenario}"
        )

    seed_everything(seed)

    net_file = NETWORK_FILES[
        grid
    ]

    route_file = ROUTE_FILES[
        key
    ]

    save_dir = (
        MODELS_V2_DIR
        / f"grid{grid}"
        / "ppo"
        / scenario
    )

    log_dir = (
        RESULTS_V2_DIR
        / "training"
        / f"grid{grid}"
        / "ppo"
        / scenario
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tls_ids, _ = (
        discover_tls_and_neighbours(
            net_file
        )
    )

    expected_agents = (
        GRID_AGENT_COUNTS[
            grid
        ]
    )

    if len(tls_ids) != expected_agents:

        raise AssertionError(
            f"Expected {expected_agents} "
            f"agents, found {len(tls_ids)}."
        )

    print()
    print("=" * 80)
    print("THESIS REVISION V2 — PPO TRAINING")
    print("=" * 80)

    print(
        f"Grid:             {grid}"
    )

    print(
        f"Scenario:         {scenario}"
    )

    print(
        f"Agents:           {len(tls_ids)}"
    )

    print(
        f"Training steps:   {total_steps}"
    )

    print(
        f"Training seed:    {seed}"
    )

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    env = TrafficSignalEnv(
        net_file=net_file,
        route_file=route_file,
        num_seconds=SIM_DURATION,
        decision_interval=DECISION_INTERVAL,
        seed=seed,
        use_gui=False,
    )

    env.action_space.seed(seed)

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {}

    for tls in tls_ids:

        dummy_env = (
            DummySingleAgentEnv()
        )

        dummy_env.action_space.seed(
            seed
        )

        model = PPO(
            "MlpPolicy",
            dummy_env,

            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,

            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,

            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,

            verbose=0,
            seed=seed,
        )

        model.set_logger(
            configure_logger(
                folder=None,
                format_strings=[],
            )
        )

        model.rollout_buffer.reset()

        models[tls] = model

    # --------------------------------------------------------
    # COMPUTATIONAL METADATA
    # --------------------------------------------------------

    buffer_bytes = {
        tls:
            rollout_buffer_bytes(
                models[tls]
                .rollout_buffer
            )
        for tls in tls_ids
    }

    total_rollout_mb = (
        sum(
            buffer_bytes.values()
        )
        / 1024.0
        / 1024.0
    )

    rollout_mb_per_agent = (
        total_rollout_mb
        / len(tls_ids)
    )

    devices = sorted(
        {
            str(model.device)
            for model in models.values()
        }
    )

    training_device = (
        ",".join(devices)
    )

    print()
    print(
        f"Training device:          "
        f"{training_device}"
    )

    print(
        f"Rollout buffer / agent:   "
        f"{rollout_mb_per_agent:.4f} MB"
    )

    print(
        f"Rollout buffers total:    "
        f"{total_rollout_mb:.4f} MB"
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    obs, info = env.reset(
        seed=seed
    )

    episode_reward = {
        tls: 0.0
        for tls in tls_ids
    }

    episode_start = {
        tls:
            np.array(
                [True],
                dtype=bool,
            )
        for tls in tls_ids
    }

    episode_count = 0

    step = 0
    buffer_pos = 0
    update_count = 0

    log_rows = []

    peak_ram_mb = (
        process_tree_rss_mb()
    )

    training_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    try:

        while step < total_steps:

            actions = {}
            values = {}
            log_probs = {}

            for tls in tls_ids:

                obs_tensor, _ = (
                    models[tls]
                    .policy
                    .obs_to_tensor(
                        obs[tls]
                    )
                )

                with th.no_grad():

                    (
                        action,
                        value,
                        log_prob,
                    ) = (
                        models[tls]
                        .policy(
                            obs_tensor
                        )
                    )

                actions[tls] = int(
                    action.item()
                )

                values[tls] = value

                log_probs[tls] = (
                    log_prob
                )

            (
                next_obs,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(
                actions
            )

            done = (
                terminated
                or truncated
            )

            for tls in tls_ids:

                models[
                    tls
                ].rollout_buffer.add(

                    obs[tls],

                    np.array(
                        [
                            actions[tls]
                        ]
                    ),

                    np.array(
                        [
                            rewards[tls]
                        ]
                    ),

                    episode_start[
                        tls
                    ],

                    values[
                        tls
                    ],

                    log_probs[
                        tls
                    ],
                )

                episode_reward[
                    tls
                ] += (
                    rewards[tls]
                )

                episode_start[
                    tls
                ] = np.array(
                    [done],
                    dtype=bool,
                )

            obs = next_obs

            step += 1
            buffer_pos += 1

            # ------------------------------------------------
            # PPO UPDATE
            # ------------------------------------------------

            if buffer_pos >= N_STEPS:

                for tls in tls_ids:

                    obs_tensor, _ = (
                        models[tls]
                        .policy
                        .obs_to_tensor(
                            obs[tls]
                        )
                    )

                    with th.no_grad():

                        last_values = (
                            models[tls]
                            .policy
                            .predict_values(
                                obs_tensor
                            )
                        )

                    models[
                        tls
                    ].rollout_buffer.compute_returns_and_advantage(

                        last_values=(
                            last_values
                        ),

                        dones=np.array(
                            [done]
                        ),
                    )

                    models[tls].train()

                    models[
                        tls
                    ].rollout_buffer.reset()

                buffer_pos = 0

                update_count += 1

            # ------------------------------------------------
            # RAM
            # ------------------------------------------------

            if (
                step
                % MEMORY_SAMPLE_INTERVAL
                == 0
                or step == total_steps
            ):

                peak_ram_mb = max(
                    peak_ram_mb,
                    process_tree_rss_mb(),
                )

            # ------------------------------------------------
            # EPISODE COMPLETE
            # ------------------------------------------------

            if done:

                episode_count += 1

                avg_reward = float(
                    np.mean(
                        list(
                            episode_reward
                            .values()
                        )
                    )
                )

                print(
                    f"[step {step}] "
                    f"episode "
                    f"{episode_count} "
                    f"finished | "
                    f"avg reward/agent="
                    f"{avg_reward:.2f} | "
                    f"updates="
                    f"{update_count}"
                )

                row = {
                    "algorithm":
                        "ppo",

                    "grid":
                        grid,

                    "scenario":
                        scenario,

                    "episode":
                        episode_count,

                    "step":
                        step,

                    "avg_reward":
                        avg_reward,

                    "ppo_updates":
                        update_count,
                }

                for tls in tls_ids:

                    row[
                        f"reward_{tls}"
                    ] = (
                        episode_reward[
                            tls
                        ]
                    )

                log_rows.append(
                    row
                )

                episode_reward = {
                    tls: 0.0
                    for tls in tls_ids
                }

                obs, info = env.reset(
                    seed=seed
                )

                for tls in tls_ids:

                    episode_start[
                        tls
                    ] = np.array(
                        [True],
                        dtype=bool,
                    )

    finally:

        env.close()

    training_runtime_s = (
        time.perf_counter()
        - training_start
    )

    peak_ram_mb = max(
        peak_ram_mb,
        process_tree_rss_mb(),
    )

    # --------------------------------------------------------
    # NOTE ABOUT PARTIAL FINAL BUFFER
    # --------------------------------------------------------

    partial_buffer_steps = (
        buffer_pos
    )

    # The original implementation trains only complete
    # 128-step PPO rollout buffers. The final partial buffer
    # is intentionally not used, preserving that design.

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model_paths = []

    for tls in tls_ids:

        path = (
            save_dir
            / (
                f"ppo_"
                f"{tls}_"
                f"{scenario}.zip"
            )
        )

        models[tls].save(
            path
        )

        model_paths.append(
            path
        )

        print(
            f"Saved {path}"
        )

    total_model_mb = (
        total_model_size_mb(
            model_paths
        )
    )

    model_mb_per_agent = (
        total_model_mb
        / len(model_paths)
    )

    # --------------------------------------------------------
    # EPISODE LOG
    # --------------------------------------------------------

    episode_log = (
        log_dir
        / "episodes.csv"
    )

    if log_rows:

        with episode_log.open(
            "w",
            newline="",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    log_rows[0]
                    .keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                log_rows
            )

    else:

        with episode_log.open(
            "w",
            newline="",
        ) as handle:

            writer = csv.writer(
                handle
            )

            writer.writerow(
                [
                    "algorithm",
                    "grid",
                    "scenario",
                    "episode",
                    "step",
                    "avg_reward",
                    "ppo_updates",
                ]
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "algorithm":
            "ppo",

        "grid":
            grid,

        "scenario":
            scenario,

        "training_seed":
            seed,

        "agents":
            len(tls_ids),

        "total_steps":
            total_steps,

        "episodes_completed":
            episode_count,

        "ppo_updates":
            update_count,

        "partial_final_buffer_steps":
            partial_buffer_steps,

        "n_steps":
            N_STEPS,

        "batch_size":
            BATCH_SIZE,

        "n_epochs":
            N_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "gamma":
            GAMMA,

        "gae_lambda":
            GAE_LAMBDA,

        "clip_range":
            CLIP_RANGE,

        "ent_coef":
            ENT_COEF,

        "training_device":
            training_device,

        "rollout_buffer_mb_per_agent":
            rollout_mb_per_agent,

        "rollout_buffer_total_mb":
            total_rollout_mb,

        "model_size_mb_per_agent":
            model_mb_per_agent,

        "model_size_total_mb":
            total_model_mb,

        "peak_process_tree_ram_mb":
            peak_ram_mb,

        "training_runtime_s":
            training_runtime_s,

        "status":
            "PASS",
    }

    summary_path = (
        log_dir
        / "training_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=summary.keys(),
        )

        writer.writeheader()

        writer.writerow(
            summary
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PPO TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Training runtime:        "
        f"{training_runtime_s:.2f}s"
    )

    print(
        f"Peak process-tree RAM:   "
        f"{peak_ram_mb:.2f} MB"
    )

    print(
        f"Rollout buffers total:   "
        f"{total_rollout_mb:.4f} MB"
    )

    print(
        f"Saved models total:      "
        f"{total_model_mb:.3f} MB"
    )

    print(
        f"Saved model per agent:   "
        f"{model_mb_per_agent:.3f} MB"
    )

    print(
        f"Episodes completed:      "
        f"{episode_count}"
    )

    print(
        f"PPO updates completed:   "
        f"{update_count}"
    )

    print(
        f"Partial final buffer:    "
        f"{partial_buffer_steps} steps"
    )

    print(
        f"Training device:         "
        f"{training_device}"
    )

    print(
        f"\nTraining summary:\n"
        f"{summary_path}"
    )

    return summary_path


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        required=True,
        choices=[
            "2x2",
            "3x3",
            "4x4",
            "5x5",
        ],
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=[
            "low",
            "medium",
            "high",
            "dynamic",
        ],
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    train(
        grid=args.grid,
        scenario=args.scenario,
        total_steps=args.total_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
