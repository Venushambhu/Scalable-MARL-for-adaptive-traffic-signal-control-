"""
PPO v2 deterministic inference validation.

Uses the current 800-step smoke models only.

Validates:
- four PPO models load
- deterministic predictions are repeatable
- valid binary actions
- complete 1800-second SUMO episode
- legal green phase at all 5-second decision boundaries
- per-agent and joint inference latency
- peak process-tree RAM
- action distribution
"""

import os
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import traci

from stable_baselines3 import PPO

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    MODELS_V2_DIR,
)

from traffic_env_v2 import TrafficSignalEnv


GRID = "2x2"
SCENARIO = "medium"

EVAL_SEED = 11

SIM_DURATION = 1800
DECISION_INTERVAL = 5

MODEL_DIR = (
    MODELS_V2_DIR
    / "grid2x2"
    / "ppo"
    / "medium"
)


def process_tree_ram_mb():

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
            pass

    return (
        total
        / 1024.0
        / 1024.0
    )


def main():

    print()
    print("=" * 72)
    print("PPO V2 INFERENCE VALIDATION")
    print("=" * 72)

    env = TrafficSignalEnv(
        net_file=NETWORK_FILES[GRID],
        route_file=ROUTE_FILES[
            (
                GRID,
                SCENARIO,
            )
        ],
        use_gui=False,
        num_seconds=SIM_DURATION,
        decision_interval=DECISION_INTERVAL,
        seed=EVAL_SEED,
    )

    tls_ids = env.tls_ids

    assert len(tls_ids) == 4

    print(
        f"Agents: {tls_ids}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = {}
    model_paths = []

    for tls in tls_ids:

        path = (
            MODEL_DIR
            / f"ppo_{tls}_{SCENARIO}.zip"
        )

        assert path.exists(), (
            f"Missing model: {path}"
        )

        models[tls] = PPO.load(
            path
        )

        model_paths.append(
            path
        )

        print(
            f"PASS: loaded {path.name}"
        )

    total_model_mb = (
        sum(
            p.stat().st_size
            for p in model_paths
        )
        / 1024.0
        / 1024.0
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    obs, info = env.reset(
        seed=EVAL_SEED
    )

    # --------------------------------------------------------
    # Repeatability
    # --------------------------------------------------------

    print()
    print(
        "Checking deterministic predictions..."
    )

    for tls in tls_ids:

        a1, _ = models[tls].predict(
            obs[tls],
            deterministic=True,
        )

        a2, _ = models[tls].predict(
            obs[tls],
            deterministic=True,
        )

        a1 = int(a1)
        a2 = int(a2)

        assert a1 == a2
        assert a1 in (0, 1)

        print(
            f"PASS: {tls} -> action {a1}"
        )

    # warm-up
    for tls in tls_ids:

        models[tls].predict(
            obs[tls],
            deterministic=True,
        )

    # --------------------------------------------------------
    # Full episode
    # --------------------------------------------------------

    agent_latency_ms = []
    joint_latency_ms = []

    action_counts = {
        tls: {
            0: 0,
            1: 0,
        }
        for tls in tls_ids
    }

    decisions = 0

    peak_ram = (
        process_tree_ram_mb()
    )

    wall_start = (
        time.perf_counter()
    )

    try:

        while True:

            actions = {}

            joint_start = (
                time.perf_counter_ns()
            )

            for tls in tls_ids:

                start = (
                    time.perf_counter_ns()
                )

                action, _ = (
                    models[tls].predict(
                        obs[tls],
                        deterministic=True,
                    )
                )

                end = (
                    time.perf_counter_ns()
                )

                agent_latency_ms.append(
                    (
                        end - start
                    )
                    / 1_000_000.0
                )

                action = int(
                    action
                )

                assert action in (
                    0,
                    1,
                )

                actions[tls] = action

                action_counts[
                    tls
                ][action] += 1

            joint_end = (
                time.perf_counter_ns()
            )

            joint_latency_ms.append(
                (
                    joint_end
                    - joint_start
                )
                / 1_000_000.0
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

            decisions += 1

            # Every decision interval must finish on green.
            for tls in tls_ids:

                phase = (
                    traci.trafficlight
                    .getPhase(tls)
                )

                assert phase in (
                    0,
                    2,
                ), (
                    f"{tls}: decision "
                    f"boundary phase={phase}"
                )

            obs = next_obs

            if decisions % 20 == 0:

                peak_ram = max(
                    peak_ram,
                    process_tree_ram_mb(),
                )

            if (
                terminated
                or truncated
            ):

                break

    finally:

        env.close()

    runtime = (
        time.perf_counter()
        - wall_start
    )

    peak_ram = max(
        peak_ram,
        process_tree_ram_mb(),
    )

    expected_decisions = (
        SIM_DURATION
        // DECISION_INTERVAL
    )

    expected_predictions = (
        expected_decisions
        * len(tls_ids)
    )

    assert decisions == 360

    assert len(
        agent_latency_ms
    ) == 1440

    mean_agent = (
        statistics.mean(
            agent_latency_ms
        )
    )

    p95_agent = float(
        np.percentile(
            agent_latency_ms,
            95,
        )
    )

    mean_joint = (
        statistics.mean(
            joint_latency_ms
        )
    )

    p95_joint = float(
        np.percentile(
            joint_latency_ms,
            95,
        )
    )

    budget_pct = (
        mean_joint
        / 5000.0
        * 100.0
    )

    print()
    print("=" * 72)
    print(
        "PASS: PPO INFERENCE VALIDATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Decisions:             "
        f"{decisions}"
    )

    print(
        f"Prediction calls:      "
        f"{len(agent_latency_ms)}"
    )

    print(
        f"Mean agent latency:    "
        f"{mean_agent:.4f} ms"
    )

    print(
        f"P95 agent latency:     "
        f"{p95_agent:.4f} ms"
    )

    print(
        f"Mean joint latency:    "
        f"{mean_joint:.4f} ms"
    )

    print(
        f"P95 joint latency:     "
        f"{p95_joint:.4f} ms"
    )

    print(
        f"Decision budget used:  "
        f"{budget_pct:.4f}%"
    )

    print(
        f"Peak RAM:              "
        f"{peak_ram:.2f} MB"
    )

    print(
        f"Model size total:      "
        f"{total_model_mb:.3f} MB"
    )

    print(
        f"Wall runtime:          "
        f"{runtime:.3f}s"
    )

    print()
    print(
        "Action counts:"
    )

    for tls in tls_ids:

        print(
            f"  {tls}: "
            f"Action0="
            f"{action_counts[tls][0]}, "
            f"Action1="
            f"{action_counts[tls][1]}"
        )

    print()
    print(
        "[PASS] deterministic inference"
    )

    print(
        "[PASS] valid binary actions"
    )

    print(
        "[PASS] 360 decision intervals"
    )

    print(
        "[PASS] 1440 prediction calls"
    )

    print(
        "[PASS] legal green phase "
        "at every decision boundary"
    )

    print()
    print(
        "NOTE: these are 800-step smoke "
        "models, not final PPO results."
    )


if __name__ == "__main__":
    main()
