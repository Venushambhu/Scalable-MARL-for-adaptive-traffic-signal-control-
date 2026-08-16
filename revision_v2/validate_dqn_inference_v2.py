"""
DQN v2 inference smoke validation.

Checks:
- all four models load
- deterministic predictions are repeatable
- actions are binary
- full 1800-second SUMO episode runs
- decision boundaries finish on green phases
- inference latency
- peak RAM
- model size

The current 800-step models are diagnostic only.
"""

import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import traci

from stable_baselines3 import DQN

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    MODELS_V2_DIR,
    RESULTS_V2_DIR,
)

from traffic_env_v2 import TrafficSignalEnv


GRID = "2x2"
SCENARIO = "medium"

TRAINING_STEPS = 800
TRAINING_SEED = 1
EVAL_SEED = 11

SIM_DURATION = 1800
DECISION_INTERVAL = 5

MODEL_DIR = (
    MODELS_V2_DIR
    / "grid2x2"
    / "dqn"
    / "medium"
)

OUTPUT_DIR = (
    RESULTS_V2_DIR
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "dqn_2x2_medium_inference_validation.csv"
)


def process_tree_ram_mb():

    root = psutil.Process(
        os.getpid()
    )

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
        / 1024
        / 1024
    )


def main():

    print()
    print("=" * 72)
    print("DQN V2 INFERENCE VALIDATION")
    print("=" * 72)

    net_file = NETWORK_FILES[
        GRID
    ]

    route_file = ROUTE_FILES[
        (
            GRID,
            SCENARIO,
        )
    ]

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = TrafficSignalEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=SIM_DURATION,
        decision_interval=DECISION_INTERVAL,
        seed=EVAL_SEED,
    )

    tls_ids = env.tls_ids

    print(
        f"Agents: {tls_ids}"
    )

    assert len(tls_ids) == 4

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = {}
    model_paths = []

    for tls in tls_ids:

        path = (
            MODEL_DIR
            / f"dqn_{tls}_{SCENARIO}.zip"
        )

        assert path.exists(), (
            f"Missing model: {path}"
        )

        models[tls] = DQN.load(
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
            path.stat().st_size
            for path in model_paths
        )
        / 1024
        / 1024
    )

    # --------------------------------------------------------
    # Start SUMO
    # --------------------------------------------------------

    obs, info = env.reset(
        seed=EVAL_SEED
    )

    # --------------------------------------------------------
    # Deterministic repeatability check
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

    # Warm-up prediction
    for tls in tls_ids:

        models[tls].predict(
            obs[tls],
            deterministic=True,
        )

    # --------------------------------------------------------
    # Full inference episode
    # --------------------------------------------------------

    prediction_latencies_ms = []
    joint_latencies_ms = []

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

                prediction_latencies_ms.append(
                    (
                        end - start
                    )
                    / 1_000_000
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
                ][
                    action
                ] += 1

            joint_end = (
                time.perf_counter_ns()
            )

            joint_latencies_ms.append(
                (
                    joint_end
                    - joint_start
                )
                / 1_000_000
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

            # Every 5-second decision interval must finish
            # on one of the legal green phases.
            for tls in tls_ids:

                phase = (
                    traci.trafficlight
                    .getPhase(tls)
                )

                assert phase in (
                    0,
                    2,
                ), (
                    f"{tls} ended decision "
                    f"interval on phase {phase}"
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

    wall_runtime = (
        time.perf_counter()
        - wall_start
    )

    peak_ram = max(
        peak_ram,
        process_tree_ram_mb(),
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_decisions = (
        SIM_DURATION
        // DECISION_INTERVAL
    )

    expected_predictions = (
        expected_decisions
        * len(tls_ids)
    )

    assert decisions == (
        expected_decisions
    )

    assert len(
        prediction_latencies_ms
    ) == expected_predictions

    mean_agent_ms = (
        statistics.mean(
            prediction_latencies_ms
        )
    )

    p95_agent_ms = float(
        np.percentile(
            prediction_latencies_ms,
            95,
        )
    )

    mean_joint_ms = (
        statistics.mean(
            joint_latencies_ms
        )
    )

    p95_joint_ms = float(
        np.percentile(
            joint_latencies_ms,
            95,
        )
    )

    budget_pct = (
        mean_joint_ms
        / (
            DECISION_INTERVAL
            * 1000
        )
        * 100
    )

    row = {
        "algorithm":
            "dqn",

        "grid":
            GRID,

        "scenario":
            SCENARIO,

        "training_steps":
            TRAINING_STEPS,

        "training_seed":
            TRAINING_SEED,

        "evaluation_seed":
            EVAL_SEED,

        "agents":
            len(tls_ids),

        "decisions":
            decisions,

        "prediction_calls":
            len(
                prediction_latencies_ms
            ),

        "mean_agent_predict_ms":
            mean_agent_ms,

        "p95_agent_predict_ms":
            p95_agent_ms,

        "mean_joint_decision_ms":
            mean_joint_ms,

        "p95_joint_decision_ms":
            p95_joint_ms,

        "decision_budget_pct":
            budget_pct,

        "peak_ram_mb":
            peak_ram,

        "model_size_total_mb":
            total_model_mb,

        "wall_runtime_s":
            wall_runtime,

        "status":
            "PASS",
    }

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=row.keys(),
        )

        writer.writeheader()

        writer.writerow(
            row
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "PASS: DQN INFERENCE VALIDATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Decisions:             "
        f"{decisions}"
    )

    print(
        f"Prediction calls:      "
        f"{len(prediction_latencies_ms)}"
    )

    print(
        f"Mean agent latency:    "
        f"{mean_agent_ms:.4f} ms"
    )

    print(
        f"P95 agent latency:     "
        f"{p95_agent_ms:.4f} ms"
    )

    print(
        f"Mean joint latency:    "
        f"{mean_joint_ms:.4f} ms"
    )

    print(
        f"P95 joint latency:     "
        f"{p95_joint_ms:.4f} ms"
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
        f"{wall_runtime:.3f}s"
    )

    print()
    print(
        "Action counts:"
    )

    for tls in tls_ids:

        print(
            f"  {tls}: "
            f"Action0={action_counts[tls][0]}, "
            f"Action1={action_counts[tls][1]}"
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
        f"CSV: {OUTPUT_CSV}"
    )

    print()
    print(
        "NOTE: performance is not being "
        "judged here because these models "
        "were trained for only 800 steps."
    )


if __name__ == "__main__":
    main()
