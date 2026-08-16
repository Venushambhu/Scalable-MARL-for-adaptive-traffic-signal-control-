"""
train_dqn_marl_v2.py

Independent multi-agent DQN training for Thesis Revision V2.

One Stable-Baselines3 DQN model is maintained per traffic-light
intersection. All agents interact with one shared SUMO simulation,
but each agent stores and learns from its own local transition stream.

Revision V2 changes
-------------------
1. Uses TrafficSignalEnvV2 with corrected traffic-signal semantics.
2. Preserves the original thesis DQN hyperparameters.
3. Explicitly seeds Python, NumPy, Gymnasium action sampling and SB3.
4. Measures:
       - training wall-clock runtime
       - peak process-tree RAM
       - allocated replay-buffer memory
       - saved model size
5. Saves only under models_v2/.
6. Logs only under results_v2/.
7. Supports short smoke training before full 50,000-step runs.

Original thesis models/results are never modified.
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

from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.utils import polyak_update

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
# ORIGINAL DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
LEARNING_STARTS = 500
BATCH_SIZE = 64
GAMMA = 0.95
TRAIN_FREQUENCY = 4
TARGET_UPDATE_INTERVAL = 250

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.70

SIM_DURATION = 1800
DECISION_INTERVAL = 5

MEMORY_SAMPLE_INTERVAL = 50


# ============================================================
# DUMMY SINGLE-AGENT ENVIRONMENT
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

        self.action_space = spaces.Discrete(
            2
        )

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

    def step(
        self,
        action,
    ):

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

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )


# ============================================================
# EPSILON SCHEDULE
# ============================================================

def linear_epsilon(
    step,
    total_steps,
    start=EPSILON_START,
    end=EPSILON_END,
    decay_fraction=EPSILON_DECAY_FRACTION,
):

    decay_steps = (
        total_steps
        * decay_fraction
    )

    if step >= decay_steps:

        return end

    return (
        start
        - (
            start
            - end
        )
        * (
            step
            / decay_steps
        )
    )


# ============================================================
# MEMORY MEASUREMENT
# ============================================================

def process_tree_rss_mb():

    try:

        root = psutil.Process(
            os.getpid()
        )

    except psutil.Error:

        return 0.0

    processes = [
        root
    ]

    try:

        processes.extend(
            root.children(
                recursive=True
            )
        )

    except psutil.Error:

        pass

    total_bytes = 0

    for process in processes:

        try:

            total_bytes += (
                process.memory_info().rss
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            continue

    return (
        total_bytes
        / 1024.0
        / 1024.0
    )


# ============================================================
# REPLAY BUFFER MEMORY
# ============================================================

def numpy_bytes(
    value,
):

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.nbytes

    return 0


def replay_buffer_bytes(
    replay_buffer,
):
    """
    Count the main NumPy arrays allocated by SB3 ReplayBuffer.
    """

    attributes = [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "timeouts",
    ]

    total = 0

    seen_ids = set()

    for attribute in attributes:

        value = getattr(
            replay_buffer,
            attribute,
            None,
        )

        if value is None:

            continue

        identity = id(
            value
        )

        if identity in seen_ids:

            continue

        seen_ids.add(
            identity
        )

        total += numpy_bytes(
            value
        )

    return total


# ============================================================
# FILE SIZE
# ============================================================

def total_model_size_mb(
    model_paths,
):

    total_bytes = sum(
        Path(path).stat().st_size
        for path in model_paths
        if Path(path).exists()
    )

    return (
        total_bytes
        / 1024.0
        / 1024.0
    )


# ============================================================
# TRAINING
# ============================================================

def train(
    grid,
    scenario,
    total_steps=50_000,
    seed=1,
):

    if (
        grid,
        scenario,
    ) not in ROUTE_FILES:

        raise KeyError(
            f"No configured route for "
            f"{grid}/{scenario}"
        )

    seed_everything(
        seed
    )

    net_file = NETWORK_FILES[
        grid
    ]

    route_file = ROUTE_FILES[
        (
            grid,
            scenario,
        )
    ]

    save_dir = (
        MODELS_V2_DIR
        / f"grid{grid}"
        / "dqn"
        / scenario
    )

    log_dir = (
        RESULTS_V2_DIR
        / "training"
        / f"grid{grid}"
        / "dqn"
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
            f"agents for {grid}, "
            f"found {len(tls_ids)}."
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "THESIS REVISION V2 — DQN TRAINING"
    )

    print(
        "=" * 80
    )

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

    print(
        f"Decision interval:{DECISION_INTERVAL}s"
    )

    print(
        "\nAgents:"
    )

    print(
        "  "
        + ", ".join(
            tls_ids
        )
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

    env.action_space.seed(
        seed
    )

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

        models[tls] = DQN(
            "MlpPolicy",
            dummy_env,

            learning_rate=(
                LEARNING_RATE
            ),

            buffer_size=(
                BUFFER_SIZE
            ),

            learning_starts=(
                LEARNING_STARTS
            ),

            batch_size=(
                BATCH_SIZE
            ),

            gamma=GAMMA,

            train_freq=1,

            target_update_interval=(
                TARGET_UPDATE_INTERVAL
            ),

            verbose=0,

            seed=seed,
        )

        models[tls].set_logger(
            configure_logger(
                folder=None,
                format_strings=[],
            )
        )

    # --------------------------------------------------------
    # REPLAY BUFFER ALLOCATION
    # --------------------------------------------------------

    replay_bytes_by_agent = {
        tls:
            replay_buffer_bytes(
                models[tls]
                .replay_buffer
            )
        for tls in tls_ids
    }

    total_replay_buffer_mb = (
        sum(
            replay_bytes_by_agent
            .values()
        )
        / 1024.0
        / 1024.0
    )

    replay_buffer_mb_per_agent = (
        total_replay_buffer_mb
        / len(tls_ids)
    )

    print(
        f"\nAllocated replay-buffer memory:"
    )

    print(
        f"  Per agent: "
        f"{replay_buffer_mb_per_agent:.3f} MB"
    )

    print(
        f"  Total:     "
        f"{total_replay_buffer_mb:.3f} MB"
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

    episode_count = 0
    step = 0

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

            epsilon = (
                linear_epsilon(
                    step,
                    total_steps,
                )
            )

            actions = {}

            for tls in tls_ids:

                if (
                    random.random()
                    < epsilon
                ):

                    actions[tls] = (
                        env.action_space
                        .sample()
                    )

                else:

                    action, _ = (
                        models[tls]
                        .predict(
                            obs[tls],
                            deterministic=True,
                        )
                    )

                    actions[tls] = int(
                        action
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
                ].replay_buffer.add(
                    obs[tls],
                    next_obs[tls],

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

                    np.array(
                        [
                            done
                        ]
                    ),

                    [{}],
                )

                episode_reward[
                    tls
                ] += rewards[
                    tls
                ]

                # Preserve original thesis
                # training schedule.
                if (
                    step
                    > LEARNING_STARTS
                    and step
                    % TRAIN_FREQUENCY
                    == 0
                ):

                    models[
                        tls
                    ].train(
                        gradient_steps=1,
                        batch_size=(
                            BATCH_SIZE
                        ),
                    )

                if (
                    step
                    % TARGET_UPDATE_INTERVAL
                    == 0
                ):

                    polyak_update(
                        models[
                            tls
                        ]
                        .q_net
                        .parameters(),

                        models[
                            tls
                        ]
                        .q_net_target
                        .parameters(),

                        models[
                            tls
                        ].tau,
                    )

            obs = next_obs

            step += 1

            # ------------------------------------------------
            # MEMORY SAMPLING
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
            # EPISODE END
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
                    f"episode {episode_count} "
                    f"finished | "
                    f"avg reward/agent="
                    f"{avg_reward:.2f} | "
                    f"epsilon="
                    f"{epsilon:.3f}"
                )

                row = {
                    "algorithm":
                        "dqn",

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

                    "epsilon":
                        epsilon,
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
    # SAVE MODELS
    # --------------------------------------------------------

    model_paths = []

    for tls in tls_ids:

        path = (
            save_dir
            / (
                f"dqn_"
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

    total_models_mb = (
        total_model_size_mb(
            model_paths
        )
    )

    model_size_mb_per_agent = (
        total_models_mb
        / len(model_paths)
    )

    # --------------------------------------------------------
    # TRAINING LOG
    # --------------------------------------------------------

    log_path = (
        log_dir
        / "episodes.csv"
    )

    if log_rows:

        with log_path.open(
            "w",
            newline="",
        ) as handle:

            writer = (
                csv.DictWriter(
                    handle,
                    fieldnames=(
                        log_rows[0]
                        .keys()
                    ),
                )
            )

            writer.writeheader()

            writer.writerows(
                log_rows
            )

    else:

        with log_path.open(
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
                    "epsilon",
                ]
            )

    # --------------------------------------------------------
    # TRAINING SUMMARY
    # --------------------------------------------------------

    summary_path = (
        log_dir
        / "training_summary.csv"
    )

    summary = {
        "algorithm":
            "dqn",

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

        "learning_rate":
            LEARNING_RATE,

        "buffer_size_per_agent":
            BUFFER_SIZE,

        "learning_starts":
            LEARNING_STARTS,

        "batch_size":
            BATCH_SIZE,

        "gamma":
            GAMMA,

        "train_frequency":
            TRAIN_FREQUENCY,

        "target_update_interval":
            TARGET_UPDATE_INTERVAL,

        "epsilon_start":
            EPSILON_START,

        "epsilon_end":
            EPSILON_END,

        "epsilon_decay_fraction":
            EPSILON_DECAY_FRACTION,

        "replay_buffer_mb_per_agent":
            replay_buffer_mb_per_agent,

        "replay_buffer_total_mb":
            total_replay_buffer_mb,

        "model_size_mb_per_agent":
            model_size_mb_per_agent,

        "model_size_total_mb":
            total_models_mb,

        "peak_process_tree_ram_mb":
            peak_ram_mb,

        "training_runtime_s":
            training_runtime_s,

        "status":
            "PASS",
    }

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

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DQN TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Training runtime:       "
        f"{training_runtime_s:.2f}s"
    )

    print(
        f"Peak process-tree RAM:  "
        f"{peak_ram_mb:.2f} MB"
    )

    print(
        f"Replay buffer total:    "
        f"{total_replay_buffer_mb:.3f} MB"
    )

    print(
        f"Saved models total:     "
        f"{total_models_mb:.3f} MB"
    )

    print(
        f"Saved model per agent:  "
        f"{model_size_mb_per_agent:.3f} MB"
    )

    print(
        f"Episodes completed:     "
        f"{episode_count}"
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
