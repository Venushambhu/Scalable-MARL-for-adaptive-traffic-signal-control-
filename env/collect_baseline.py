"""
collect_baseline.py (Phase 2: generalized for any grid size)

Runs the fixed-time baseline (SUMO's native static signal program) for
any grid size and logs per-step + per-trip metrics, matching the
metric set used for DQN/PPO evaluation.

Usage:
    python collect_baseline.py --grid 2x2 --scenario low --seed 1
"""

import os
import sys
import argparse
import csv
import xml.etree.ElementTree as ET

import traci

sys.path.append(".")
from traffic_env import discover_tls_and_neighbours

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

LOG_STEP = 5


def run_baseline(grid, scenario, seed, num_seconds=1800):
    net_file = f"../network/grid{grid}/grid{grid}.net.xml"
    route_file = f"../routes/grid{grid}/{scenario}_demand.rou.xml"
    out_dir = f"../results/grid{grid}"
    os.makedirs(out_dir, exist_ok=True)

    tls_ids, _ = discover_tls_and_neighbours(net_file)

    step_out_path = os.path.join(out_dir, f"fixedtime_{scenario}_seed{seed}_steps.csv")
    trip_out_path = os.path.join(out_dir, f"fixedtime_{scenario}_seed{seed}_tripinfo.xml")
    trip_summary_path = os.path.join(out_dir, f"fixedtime_{scenario}_seed{seed}_summary.csv")

    sumo_cmd = [
        "sumo",
        "-n", net_file,
        "-r", route_file,
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "300",
        "--seed", str(seed),
        "--tripinfo-output", trip_out_path,
    ]
    traci.start(sumo_cmd)

    controlled_lanes = {tls: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls))) for tls in tls_ids}

    rows = []
    step = 0
    while step < num_seconds:
        traci.simulationStep()
        step += 1

        if step % LOG_STEP == 0:
            for tls in tls_ids:
                lanes = controlled_lanes[tls]
                queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                waiting = sum(traci.lane.getWaitingTime(l) for l in lanes)
                vehicles = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
                rows.append({
                    "controller": "fixedtime", "grid": grid, "scenario": scenario,
                    "seed": seed, "time": step, "intersection": tls,
                    "queue_length": queue, "waiting_time": waiting, "vehicle_count": vehicles,
                })

    traci.close()

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
        writer.writerow(["fixedtime", grid, scenario, seed, n_completed,
                          round(avg_duration, 2), round(avg_waiting, 2), round(avg_timeloss, 2)])

    print(f"Saved trip summary to {trip_summary_path}")
    print(f"  Completed trips: {n_completed}, avg travel time: {avg_duration:.2f}s, avg waiting: {avg_waiting:.2f}s")

    return step_out_path, trip_summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="2x2", choices=["2x2", "3x3", "4x4", "5x5"])
    parser.add_argument("--scenario", required=True, choices=["low", "medium", "high", "dynamic"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_baseline(args.grid, args.scenario, args.seed)
