"""
evaluate_trained.py (Phase 2: generalized for any grid size)

Loads trained models for a given grid/controller/scenario and runs
them deterministically, logging the same metrics as the baseline.

Usage:
    python evaluate_trained.py --grid 2x2 --controller dqn --scenario low --seed 11
"""

import os
import sys
import argparse
import csv
import xml.etree.ElementTree as ET

sys.path.append(".")
from traffic_env import TrafficSignalEnv, discover_tls_and_neighbours
import traci

from stable_baselines3 import DQN, PPO

LOG_STEP = 5


def load_models(grid, controller, scenario, tls_ids):
    algo_cls = DQN if controller == "dqn" else PPO
    model_dir = f"../models/grid{grid}"
    models = {}
    for tls in tls_ids:
        path = os.path.join(model_dir, f"{controller}_{tls}_{scenario}.zip")
        models[tls] = algo_cls.load(path)
    return models


def run_evaluation(grid, controller, scenario, seed):
    net_file = f"../network/grid{grid}/grid{grid}.net.xml"
    route_file = f"../routes/grid{grid}/{scenario}_demand.rou.xml"
    out_dir = f"../results/grid{grid}"
    os.makedirs(out_dir, exist_ok=True)

    tls_ids, _ = discover_tls_and_neighbours(net_file)

    step_out_path = os.path.join(out_dir, f"{controller}_{scenario}_seed{seed}_steps.csv")
    trip_out_path = os.path.join(out_dir, f"{controller}_{scenario}_seed{seed}_tripinfo.xml")
    trip_summary_path = os.path.join(out_dir, f"{controller}_{scenario}_seed{seed}_summary.csv")

    models = load_models(grid, controller, scenario, tls_ids)

    env = TrafficSignalEnv(net_file, route_file, num_seconds=1800,
                            decision_interval=5, seed=seed,
                            tripinfo_output=trip_out_path)

    obs, info = env.reset()
    rows = []
    sim_time = 0

    while True:
        actions = {}
        for tls in tls_ids:
            action, _ = models[tls].predict(obs[tls], deterministic=True)
            actions[tls] = int(action)

        obs, rewards, terminated, truncated, info = env.step(actions)
        sim_time += env.decision_interval

        if sim_time % LOG_STEP == 0:
            for tls in tls_ids:
                lanes = env.controlled_lanes[tls]
                queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                waiting = sum(traci.lane.getWaitingTime(l) for l in lanes)
                vehicles = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
                rows.append({
                    "controller": controller, "grid": grid, "scenario": scenario,
                    "seed": seed, "time": sim_time, "intersection": tls,
                    "queue_length": queue, "waiting_time": waiting, "vehicle_count": vehicles,
                })

        if terminated or truncated:
            break

    env.close()

    with open(step_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} step-level rows to {step_out_path}")

    tree = ET.parse(trip_out_path)
    trips = tree.getroot().findall("tripinfo")
    n_completed = len(trips)
    if n_completed > 0:
        avg_duration = sum(float(t.get("duration")) for t in trips) / n_completed
        avg_waiting = sum(float(t.get("waitingTime")) for t in trips) / n_completed
        avg_timeloss = sum(float(t.get("timeLoss")) for t in trips) / n_completed
    else:
        avg_duration = avg_waiting = avg_timeloss = 0.0

    with open(trip_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", "grid", "scenario", "seed", "completed_trips",
                          "avg_travel_time", "avg_waiting_time", "avg_time_loss"])
        writer.writerow([controller, grid, scenario, seed, n_completed,
                          round(avg_duration, 2), round(avg_waiting, 2), round(avg_timeloss, 2)])

    print(f"Saved trip summary to {trip_summary_path}")
    print(f"  Completed trips: {n_completed}, avg travel time: {avg_duration:.2f}s, avg waiting: {avg_waiting:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="2x2", choices=["2x2", "3x3", "4x4", "5x5"])
    parser.add_argument("--controller", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--scenario", required=True, choices=["low", "medium", "high", "dynamic"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_evaluation(args.grid, args.controller, args.scenario, args.seed)
