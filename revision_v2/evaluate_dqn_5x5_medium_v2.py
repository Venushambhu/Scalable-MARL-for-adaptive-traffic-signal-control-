"""
evaluate_dqn_v2.py

Final deterministic evaluation of trained DQN Revision V2 models.

Preserves the original thesis metric definitions:
- completed_trips = number of SUMO tripinfo records
- avg_travel_time = mean tripinfo.duration
- avg_waiting_time = mean tripinfo.waitingTime
- avg_time_loss = mean tripinfo.timeLoss
- queue_length = halted vehicles per controlled intersection,
                 sampled every 5 seconds

Adds:
- completion rate
- teleports
- inference latency
- peak RAM
- model size

Current configuration:
    grid     = 2x2
    scenario = medium
    seeds    = 11,12,13,14,15
"""

import csv
import os
import statistics
import time
import xml.etree.ElementTree as ET
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

from traffic_env_v2 import (
    TrafficSignalEnv,
    discover_tls_and_neighbours,
)


GRID = "5x5"
SCENARIO = "medium"

SEEDS = [11, 12, 13, 14, 15]

SIM_DURATION = 1800
DECISION_INTERVAL = 5

MODEL_DIR = (
    MODELS_V2_DIR
    / f"grid{GRID}"
    / "dqn"
    / SCENARIO
)

OUTPUT_DIR = (
    RESULTS_V2_DIR
    / "learned"
    / f"grid{GRID}"
    / "dqn"
    / SCENARIO
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_CSV = (
    OUTPUT_DIR
    / "dqn_medium_all_seeds.csv"
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


def count_loaded_vehicles(route_file):

    root = (
        ET.parse(route_file)
        .getroot()
    )

    return len(
        root.findall(".//vehicle")
    )


def analyse_tripinfo(path):

    root = (
        ET.parse(path)
        .getroot()
    )

    trips = root.findall(
        "tripinfo"
    )

    n = len(trips)

    if n == 0:
        return 0, 0.0, 0.0, 0.0

    travel = [
        float(t.get("duration"))
        for t in trips
    ]

    waiting = [
        float(t.get("waitingTime"))
        for t in trips
    ]

    time_loss = [
        float(t.get("timeLoss"))
        for t in trips
    ]

    return (
        n,
        statistics.mean(travel),
        statistics.mean(waiting),
        statistics.mean(time_loss),
    )


def load_models(tls_ids):

    models = {}
    paths = []

    for tls in tls_ids:

        path = (
            MODEL_DIR
            / f"dqn_{tls}_{SCENARIO}.zip"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing model: {path}"
            )

        models[tls] = DQN.load(
            path
        )

        paths.append(path)

    return models, paths


def run_seed(seed):

    print()
    print("=" * 78)
    print(
        f"DQN V2 | {GRID} | "
        f"{SCENARIO} | seed {seed}"
    )
    print("=" * 78)

    net_file = NETWORK_FILES[GRID]

    route_file = ROUTE_FILES[
        (
            GRID,
            SCENARIO,
        )
    ]

    loaded = count_loaded_vehicles(
        route_file
    )

    tls_ids, _ = (
        discover_tls_and_neighbours(
            net_file
        )
    )

    models, model_paths = (
        load_models(
            tls_ids
        )
    )

    model_total_mb = (
        sum(
            p.stat().st_size
            for p in model_paths
        )
        / 1024.0
        / 1024.0
    )

    tripinfo_file = (
        OUTPUT_DIR
        / f"dqn_medium_seed{seed}_tripinfo.xml"
    )

    step_file = (
        OUTPUT_DIR
        / f"dqn_medium_seed{seed}_steps.csv"
    )

    if tripinfo_file.exists():
        tripinfo_file.unlink()

    env = TrafficSignalEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=SIM_DURATION,
        decision_interval=DECISION_INTERVAL,
        seed=seed,
        tripinfo_output=tripinfo_file,
    )

    obs, info = env.reset(
        seed=seed
    )

    controlled_lanes = (
        env.controlled_lanes
    )

    prediction_ms = []
    joint_ms = []

    action_counts = {
        tls: {0: 0, 1: 0}
        for tls in tls_ids
    }

    queue_sum = 0.0
    queue_samples = 0
    max_queue = 0

    departed = 0
    arrived = 0
    teleports = 0
    teleports_ended = 0

    peak_ram = (
        process_tree_ram_mb()
    )

    wall_start = (
        time.perf_counter()
    )

    fieldnames = [
        "controller",
        "grid",
        "scenario",
        "seed",
        "time",
        "intersection",
        "queue_length",
        "waiting_time",
        "vehicle_count",
    ]

    with step_file.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        decision = 0

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

                    prediction_ms.append(
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

                joint_ms.append(
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

                decision += 1

                sim_time = (
                    decision
                    * DECISION_INTERVAL
                )

                departed += int(
                    info[
                        "departed_vehicles"
                    ]
                )

                arrived += int(
                    info[
                        "arrived_vehicles"
                    ]
                )

                teleports += int(
                    info[
                        "teleports_started"
                    ]
                )

                teleports_ended += int(
                    info[
                        "teleports_ended"
                    ]
                )

                for tls in tls_ids:

                    phase = (
                        traci.trafficlight
                        .getPhase(tls)
                    )

                    if phase not in (
                        0,
                        2,
                    ):
                        raise AssertionError(
                            f"{tls}: illegal "
                            f"decision-boundary "
                            f"phase {phase}"
                        )

                    lanes = (
                        controlled_lanes[
                            tls
                        ]
                    )

                    queue = sum(
                        traci.lane
                        .getLastStepHaltingNumber(
                            lane
                        )
                        for lane in lanes
                    )

                    waiting = sum(
                        traci.lane
                        .getWaitingTime(
                            lane
                        )
                        for lane in lanes
                    )

                    vehicles = sum(
                        traci.lane
                        .getLastStepVehicleNumber(
                            lane
                        )
                        for lane in lanes
                    )

                    writer.writerow(
                        {
                            "controller":
                                "dqn",

                            "grid":
                                GRID,

                            "scenario":
                                SCENARIO,

                            "seed":
                                seed,

                            "time":
                                sim_time,

                            "intersection":
                                tls,

                            "queue_length":
                                queue,

                            "waiting_time":
                                waiting,

                            "vehicle_count":
                                vehicles,
                        }
                    )

                    queue_sum += queue

                    queue_samples += 1

                    max_queue = max(
                        max_queue,
                        queue,
                    )

                obs = next_obs

                if decision % 20 == 0:

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

    (
        completed,
        avg_travel,
        avg_waiting,
        avg_time_loss,
    ) = analyse_tripinfo(
        tripinfo_file
    )

    if completed != arrived:

        raise AssertionError(
            f"Tripinfo/TraCI mismatch: "
            f"{completed} vs {arrived}"
        )

    completion_rate = (
        100.0
        * completed
        / loaded
    )

    mean_queue = (
        queue_sum
        / queue_samples
    )

    mean_agent_ms = (
        statistics.mean(
            prediction_ms
        )
    )

    p95_agent_ms = float(
        np.percentile(
            prediction_ms,
            95,
        )
    )

    mean_joint_ms = (
        statistics.mean(
            joint_ms
        )
    )

    p95_joint_ms = float(
        np.percentile(
            joint_ms,
            95,
        )
    )

    row = {
        "controller":
            "dqn",

        "grid":
            GRID,

        "scenario":
            SCENARIO,

        "seed":
            seed,

        "loaded_vehicles":
            loaded,

        "departed_vehicles":
            departed,

        "completed_trips":
            completed,

        "completion_rate_pct":
            completion_rate,

        "avg_travel_time":
            avg_travel,

        "avg_waiting_time":
            avg_waiting,

        "avg_time_loss":
            avg_time_loss,

        "mean_queue_length":
            mean_queue,

        "max_queue_length":
            max_queue,

        "teleports_started":
            teleports,

        "teleports_ended":
            teleports_ended,

        "mean_agent_predict_ms":
            mean_agent_ms,

        "p95_agent_predict_ms":
            p95_agent_ms,

        "mean_joint_decision_ms":
            mean_joint_ms,

        "p95_joint_decision_ms":
            p95_joint_ms,

        "peak_process_tree_ram_mb":
            peak_ram,

        "model_size_total_mb":
            model_total_mb,

        "wall_runtime_s":
            wall_runtime,

        "status":
            "PASS",
    }

    print(
        f"Completed:       "
        f"{completed}/{loaded} "
        f"({completion_rate:.2f}%)"
    )

    print(
        f"Travel:          "
        f"{avg_travel:.2f}s"
    )

    print(
        f"Waiting:         "
        f"{avg_waiting:.2f}s"
    )

    print(
        f"Queue:           "
        f"{mean_queue:.2f}"
    )

    print(
        f"Teleports:       "
        f"{teleports}"
    )

    print(
        f"Joint latency:   "
        f"{mean_joint_ms:.4f} ms"
    )

    print(
        "Actions:"
    )

    for tls in tls_ids:

        print(
            f"  {tls}: "
            f"A0={action_counts[tls][0]}, "
            f"A1={action_counts[tls][1]}"
        )

    return row


def main():

    rows = []

    for seed in SEEDS:

        rows.append(
            run_seed(seed)
        )

    with MASTER_CSV.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("=" * 78)
    print("DQN V2 — FIVE-SEED SUMMARY")
    print("=" * 78)

    metrics = [
        "completion_rate_pct",
        "avg_travel_time",
        "avg_waiting_time",
        "avg_time_loss",
        "mean_queue_length",
        "teleports_started",
        "mean_joint_decision_ms",
        "peak_process_tree_ram_mb",
    ]

    for metric in metrics:

        values = [
            float(
                row[metric]
            )
            for row in rows
        ]

        mean = statistics.mean(
            values
        )

        std = (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        )

        print(
            f"{metric:<30} "
            f"{mean:>10.3f} "
            f"+/- {std:.3f}"
        )

    print()
    print(
        f"CSV: {MASTER_CSV}"
    )

    print()
    print(
        "PASS: 5/5 deterministic "
        "DQN evaluation seeds completed."
    )


if __name__ == "__main__":
    main()
