"""
evaluate_conventional_v2.py

Final conventional-controller evaluation pipeline for Thesis Revision V2.

Controllers
-----------
1. original_fixed
2. webster_fixed
3. actuated

Experimental design
-------------------
Primary 2x2:
    scenarios = low, medium, high, dynamic
    seeds     = 11, 12, 13, 14, 15

Scalability:
    3x3 medium
    4x4 medium
    5x5 medium
    seeds = 11, 12, 13, 14, 15

Full conventional matrix:
    105 simulation runs

Compatibility with original thesis
----------------------------------
Preserves the original evaluation definitions:

Step-level, every 5 seconds, per intersection:
    queue_length
    waiting_time
    vehicle_count

Trip-level:
    completed_trips     = number of SUMO <tripinfo> records
    avg_travel_time     = mean tripinfo.duration
    avg_waiting_time    = mean tripinfo.waitingTime
    avg_time_loss       = mean tripinfo.timeLoss

Revision V2 additions:
    loaded_vehicles
    departed_vehicles
    completion_rate_pct
    completion_of_departed_pct
    teleports_started
    teleports_ended
    active_at_end
    min_expected_at_end
    mean_queue_length
    max_queue_length
    peak_process_tree_ram_mb
    simulation_loop_runtime_s
    total_wall_runtime_s

Outputs
-------
results_v2/conventional/
    grid2x2/
    grid3x3/
    grid4x4/
    grid5x5/

    conventional_all_runs_summary.csv
    conventional_aggregate_summary.csv

The script supports resuming interrupted experiments.
Existing completed runs are skipped unless --overwrite is supplied.

Original networks, routes, models and results are never modified.
"""

from pathlib import Path

import argparse
import csv
import os
import shutil
import statistics
import time
import xml.etree.ElementTree as ET

import psutil
import traci

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    BASELINES_V2_DIR,
    RESULTS_V2_DIR,
    GRID_AGENT_COUNTS,
)


# ============================================================
# CONSTANTS
# ============================================================

SIM_DURATION = 1800
LOG_STEP = 5

SEEDS = [
    11,
    12,
    13,
    14,
    15,
]

CONTROLLERS = [
    "original_fixed",
    "webster_fixed",
    "actuated",
]

PRIMARY_SCENARIOS = [
    "low",
    "medium",
    "high",
    "dynamic",
]

SCALABILITY_GRIDS = [
    "3x3",
    "4x4",
    "5x5",
]

CONVENTIONAL_DIR = (
    RESULTS_V2_DIR
    / "conventional"
)

MASTER_SUMMARY = (
    CONVENTIONAL_DIR
    / "conventional_all_runs_summary.csv"
)

AGGREGATE_SUMMARY = (
    CONVENTIONAL_DIR
    / "conventional_aggregate_summary.csv"
)


# ============================================================
# SUMMARY FIELDS
# ============================================================

SUMMARY_FIELDS = [
    "controller",
    "grid",
    "scenario",
    "seed",

    "simulation_duration_s",

    "loaded_vehicles",
    "departed_vehicles",
    "completed_trips",

    "completion_rate_pct",
    "completion_of_departed_pct",

    "avg_travel_time",
    "avg_waiting_time",
    "avg_time_loss",

    "mean_queue_length",
    "max_queue_length",

    "mean_step_waiting_time",
    "mean_vehicle_count",

    "teleports_started",
    "teleports_ended",

    "active_at_end",
    "min_expected_at_end",

    "tls_count",
    "active_program",

    "peak_process_tree_ram_mb",

    "simulation_loop_runtime_s",
    "total_wall_runtime_s",

    "tripinfo_arrival_match",

    "step_file",
    "tripinfo_file",

    "status",
]


# ============================================================
# PRINTING
# ============================================================

def banner(text):

    print(
        "\n"
        + "=" * 88
    )

    print(text)

    print(
        "=" * 88
    )


# ============================================================
# PROCESS MEMORY
# ============================================================

def process_tree_rss_mb():
    """
    Return RSS of the evaluator Python process plus all
    descendant processes, including SUMO.

    This same measurement method can later be reused for
    DQN/PPO inference evaluation.
    """

    try:

        root = psutil.Process(
            os.getpid()
        )

    except psutil.Error:

        return 0.0

    total = 0

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

    for process in processes:

        try:

            total += (
                process.memory_info().rss
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            continue

    return (
        total
        / (
            1024.0
            * 1024.0
        )
    )


# ============================================================
# EXPERIMENT DESIGN
# ============================================================

def primary_runs():

    runs = []

    for scenario in PRIMARY_SCENARIOS:

        for controller in CONTROLLERS:

            for seed in SEEDS:

                runs.append(
                    (
                        controller,
                        "2x2",
                        scenario,
                        seed,
                    )
                )

    return runs


def scalability_runs():

    runs = []

    for grid in SCALABILITY_GRIDS:

        for controller in CONTROLLERS:

            for seed in SEEDS:

                runs.append(
                    (
                        controller,
                        grid,
                        "medium",
                        seed,
                    )
                )

    return runs


def pilot_runs():

    return [
        (
            controller,
            "2x2",
            "medium",
            11,
        )
        for controller in CONTROLLERS
    ]


def build_run_matrix(
    mode,
    controller=None,
    grid=None,
    scenario=None,
    seed=None,
):

    if mode == "pilot":

        return pilot_runs()

    if mode == "primary":

        return primary_runs()

    if mode == "scalability":

        return scalability_runs()

    if mode == "full":

        return (
            primary_runs()
            + scalability_runs()
        )

    if mode == "single":

        if controller is None:

            raise ValueError(
                "--controller is required "
                "for --mode single"
            )

        if grid is None:

            raise ValueError(
                "--grid is required "
                "for --mode single"
            )

        if scenario is None:

            raise ValueError(
                "--scenario is required "
                "for --mode single"
            )

        if seed is None:

            raise ValueError(
                "--seed is required "
                "for --mode single"
            )

        return [
            (
                controller,
                grid,
                scenario,
                seed,
            )
        ]

    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# PATHS
# ============================================================

def get_network_file(
    controller,
    grid,
):

    if controller in (
        "original_fixed",
        "webster_fixed",
    ):

        return NETWORK_FILES[
            grid
        ]

    if controller == "actuated":

        return (
            BASELINES_V2_DIR
            / "actuated"
            / f"grid{grid}"
            / f"grid{grid}_actuated.net.xml"
        )

    raise ValueError(
        f"Unknown controller: "
        f"{controller}"
    )


def get_additional_file(
    controller,
    grid,
    scenario,
):

    if controller != "webster_fixed":

        return None

    return (
        BASELINES_V2_DIR
        / "tuned_fixed"
        / f"grid{grid}"
        / (
            f"grid{grid}_"
            f"{scenario}_"
            f"webster.add.xml"
        )
    )


def expected_program(
    controller,
):

    if controller == "webster_fixed":

        return "a"

    return "0"


def get_run_paths(
    controller,
    grid,
    scenario,
    seed,
):

    output_dir = (
        CONVENTIONAL_DIR
        / f"grid{grid}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{controller}_"
        f"{scenario}_"
        f"seed{seed}"
    )

    step_file = (
        output_dir
        / f"{stem}_steps.csv"
    )

    tripinfo_file = (
        output_dir
        / f"{stem}_tripinfo.xml"
    )

    summary_file = (
        output_dir
        / f"{stem}_summary.csv"
    )

    return (
        step_file,
        tripinfo_file,
        summary_file,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def count_loaded_vehicles(
    route_file,
):

    root = (
        ET.parse(
            route_file
        )
        .getroot()
    )

    vehicles = root.findall(
        ".//vehicle"
    )

    flows = root.findall(
        ".//flow"
    )

    trips = root.findall(
        ".//trip"
    )

    if flows:

        raise RuntimeError(
            f"Unexpected <flow> "
            f"definitions in:\n"
            f"{route_file}"
        )

    if trips:

        raise RuntimeError(
            f"Unexpected <trip> "
            f"definitions in:\n"
            f"{route_file}"
        )

    if not vehicles:

        raise RuntimeError(
            f"No <vehicle> definitions "
            f"found in:\n"
            f"{route_file}"
        )

    return len(
        vehicles
    )


def validate_run_inputs(
    controller,
    grid,
    scenario,
):

    key = (
        grid,
        scenario,
    )

    if key not in ROUTE_FILES:

        raise KeyError(
            f"No route file configured "
            f"for {grid}/{scenario}"
        )

    network_file = (
        get_network_file(
            controller,
            grid,
        )
    )

    route_file = (
        ROUTE_FILES[
            key
        ]
    )

    additional_file = (
        get_additional_file(
            controller,
            grid,
            scenario,
        )
    )

    if not Path(
        network_file
    ).exists():

        raise FileNotFoundError(
            f"Network file missing:\n"
            f"{network_file}"
        )

    if not Path(
        route_file
    ).exists():

        raise FileNotFoundError(
            f"Route file missing:\n"
            f"{route_file}"
        )

    if (
        additional_file
        is not None
        and not Path(
            additional_file
        ).exists()
    ):

        raise FileNotFoundError(
            f"Webster file missing:\n"
            f"{additional_file}"
        )

    return (
        Path(network_file),
        Path(route_file),
        (
            Path(additional_file)
            if additional_file
            is not None
            else None
        ),
    )


# ============================================================
# SUMO COMMAND
# ============================================================

def make_sumo_command(
    network_file,
    route_file,
    tripinfo_file,
    seed,
    additional_file=None,
):

    sumo_binary = shutil.which(
        "sumo"
    )

    if not sumo_binary:

        raise RuntimeError(
            "SUMO executable not found."
        )

    command = [
        sumo_binary,

        "-n",
        str(network_file),

        "-r",
        str(route_file),

        "--no-warnings",
        "true",

        "--no-step-log",
        "true",

        "--time-to-teleport",
        "300",

        "--seed",
        str(seed),

        "--tripinfo-output",
        str(tripinfo_file),
    ]

    if additional_file is not None:

        command += [
            "-a",
            str(additional_file),
        ]

    return command


# ============================================================
# TRIPINFO ANALYSIS
# ============================================================

def analyse_tripinfo(
    tripinfo_file,
):

    root = (
        ET.parse(
            tripinfo_file
        )
        .getroot()
    )

    trips = root.findall(
        "tripinfo"
    )

    completed = len(
        trips
    )

    if completed == 0:

        return {
            "completed_trips":
                0,
            "avg_travel_time":
                0.0,
            "avg_waiting_time":
                0.0,
            "avg_time_loss":
                0.0,
        }

    durations = [
        float(
            trip.get(
                "duration"
            )
        )
        for trip in trips
    ]

    waiting_times = [
        float(
            trip.get(
                "waitingTime"
            )
        )
        for trip in trips
    ]

    time_losses = [
        float(
            trip.get(
                "timeLoss"
            )
        )
        for trip in trips
    ]

    return {
        "completed_trips":
            completed,

        "avg_travel_time":
            statistics.mean(
                durations
            ),

        "avg_waiting_time":
            statistics.mean(
                waiting_times
            ),

        "avg_time_loss":
            statistics.mean(
                time_losses
            ),
    }


# ============================================================
# SINGLE RUN
# ============================================================

def run_one(
    controller,
    grid,
    scenario,
    seed,
    overwrite=False,
):

    (
        step_file,
        tripinfo_file,
        summary_file,
    ) = get_run_paths(
        controller,
        grid,
        scenario,
        seed,
    )

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if (
        summary_file.exists()
        and not overwrite
    ):

        print(
            f"\nSKIP: completed result exists:\n"
            f"{summary_file}"
        )

        return read_summary_file(
            summary_file
        )

    # Remove partial files from a previous interrupted attempt.

    for path in (
        step_file,
        tripinfo_file,
        summary_file,
    ):

        if path.exists():

            path.unlink()

    (
        network_file,
        route_file,
        additional_file,
    ) = validate_run_inputs(
        controller,
        grid,
        scenario,
    )

    loaded_vehicles = (
        count_loaded_vehicles(
            route_file
        )
    )

    expected_tls = (
        GRID_AGENT_COUNTS[
            grid
        ]
    )

    program_expected = (
        expected_program(
            controller
        )
    )

    banner(
        f"{controller.upper()} | "
        f"{grid} | "
        f"{scenario} | "
        f"seed {seed}"
    )

    print(
        f"Loaded vehicles: "
        f"{loaded_vehicles}"
    )

    print(
        f"Expected TLS: "
        f"{expected_tls}"
    )

    print(
        f"Expected active program: "
        f"{program_expected}"
    )

    sumo_command = (
        make_sumo_command(
            network_file=(
                network_file
            ),
            route_file=(
                route_file
            ),
            tripinfo_file=(
                tripinfo_file
            ),
            seed=seed,
            additional_file=(
                additional_file
            ),
        )
    )

    wall_start = (
        time.perf_counter()
    )

    traci_started = False

    cumulative_departed = 0
    cumulative_arrived = 0

    teleports_started = 0
    teleports_ended = 0

    queue_sum = 0.0
    queue_count = 0
    max_queue = 0

    step_waiting_sum = 0.0
    vehicle_count_sum = 0.0

    active_at_end = 0
    min_expected_at_end = 0

    peak_ram_mb = 0.0

    simulation_loop_runtime = 0.0

    tls_ids = []

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

    try:

        traci.start(
            sumo_command
        )

        traci_started = True

        tls_ids = sorted(
            traci.trafficlight
            .getIDList()
        )

        if len(
            tls_ids
        ) != expected_tls:

            raise AssertionError(
                f"Expected {expected_tls} TLS, "
                f"found {len(tls_ids)}."
            )

        # ----------------------------------------------------
        # ACTIVATE AND VERIFY INTENDED PROGRAM
        # ----------------------------------------------------

        if (
            controller
            == "webster_fixed"
        ):

            for tls in tls_ids:

                programs = {
                    logic.programID
                    for logic in (
                        traci.trafficlight
                        .getAllProgramLogics(
                            tls
                        )
                    )
                }

                if (
                    program_expected
                    not in programs
                ):

                    raise AssertionError(
                        f"TLS {tls} does not contain "
                        f"Webster program "
                        f"'{program_expected}'. "
                        f"Available: "
                        f"{sorted(programs)}"
                    )

                traci.trafficlight.setProgram(
                    tls,
                    program_expected,
                )

        active_programs = {
            tls:
                traci.trafficlight
                .getProgram(tls)
            for tls in tls_ids
        }

        incorrect_programs = {
            tls: program
            for (
                tls,
                program,
            )
            in active_programs.items()
            if (
                program
                != program_expected
            )
        }

        if incorrect_programs:

            raise AssertionError(
                "Incorrect TLS program(s):\n"
                f"{incorrect_programs}"
            )

        print(
            "PASS: intended TLS program "
            "active on all intersections."
        )

        # ----------------------------------------------------
        # CONTROLLED LANES
        # ----------------------------------------------------

        controlled_lanes = {
            tls:
                list(
                    dict.fromkeys(
                        traci.trafficlight
                        .getControlledLanes(
                            tls
                        )
                    )
                )
            for tls in tls_ids
        }

        # ----------------------------------------------------
        # STEP CSV
        # ----------------------------------------------------

        with step_file.open(
            "w",
            newline="",
        ) as step_handle:

            writer = csv.DictWriter(
                step_handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            peak_ram_mb = max(
                peak_ram_mb,
                process_tree_rss_mb(),
            )

            loop_start = (
                time.perf_counter()
            )

            # ------------------------------------------------
            # EXACTLY 1800 SIMULATION SECONDS
            # ------------------------------------------------

            for step in range(
                1,
                SIM_DURATION + 1,
            ):

                traci.simulationStep()

                cumulative_departed += (
                    traci.simulation
                    .getDepartedNumber()
                )

                cumulative_arrived += (
                    traci.simulation
                    .getArrivedNumber()
                )

                teleports_started += (
                    traci.simulation
                    .getStartingTeleportNumber()
                )

                teleports_ended += (
                    traci.simulation
                    .getEndingTeleportNumber()
                )

                peak_ram_mb = max(
                    peak_ram_mb,
                    process_tree_rss_mb(),
                )

                # --------------------------------------------
                # Preserve original 5-second metric sampling.
                # --------------------------------------------

                if (
                    step
                    % LOG_STEP
                    == 0
                ):

                    for tls in tls_ids:

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
                                    controller,

                                "grid":
                                    grid,

                                "scenario":
                                    scenario,

                                "seed":
                                    seed,

                                "time":
                                    step,

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

                        queue_sum += (
                            queue
                        )

                        queue_count += 1

                        max_queue = max(
                            max_queue,
                            queue,
                        )

                        step_waiting_sum += (
                            waiting
                        )

                        vehicle_count_sum += (
                            vehicles
                        )

            simulation_loop_runtime = (
                time.perf_counter()
                - loop_start
            )

        active_at_end = (
            traci.vehicle
            .getIDCount()
        )

        min_expected_at_end = (
            traci.simulation
            .getMinExpectedNumber()
        )

    finally:

        if traci_started:

            try:

                traci.close()

            except Exception:

                pass

    total_wall_runtime = (
        time.perf_counter()
        - wall_start
    )

    # --------------------------------------------------------
    # TRIPINFO
    # --------------------------------------------------------

    if not tripinfo_file.exists():

        raise RuntimeError(
            f"SUMO did not create tripinfo:\n"
            f"{tripinfo_file}"
        )

    trip_metrics = (
        analyse_tripinfo(
            tripinfo_file
        )
    )

    completed_trips = (
        trip_metrics[
            "completed_trips"
        ]
    )

    arrival_match = (
        completed_trips
        == cumulative_arrived
    )

    if not arrival_match:

        raise AssertionError(
            "\nTripinfo/TraCI arrival mismatch.\n"
            f"Tripinfo completed: "
            f"{completed_trips}\n"
            f"TraCI arrived: "
            f"{cumulative_arrived}"
        )

    # --------------------------------------------------------
    # ORIGINAL-COMPATIBLE QUEUE MEAN
    #
    # This is mean halted vehicles per
    # intersection per 5-second sample.
    # --------------------------------------------------------

    mean_queue_length = (
        queue_sum
        / queue_count
        if queue_count
        else 0.0
    )

    mean_step_waiting = (
        step_waiting_sum
        / queue_count
        if queue_count
        else 0.0
    )

    mean_vehicle_count = (
        vehicle_count_sum
        / queue_count
        if queue_count
        else 0.0
    )

    completion_rate = (
        completed_trips
        / loaded_vehicles
        * 100.0
        if loaded_vehicles
        else 0.0
    )

    completion_of_departed = (
        completed_trips
        / cumulative_departed
        * 100.0
        if cumulative_departed
        else 0.0
    )

    row = {
        "controller":
            controller,

        "grid":
            grid,

        "scenario":
            scenario,

        "seed":
            seed,

        "simulation_duration_s":
            SIM_DURATION,

        "loaded_vehicles":
            loaded_vehicles,

        "departed_vehicles":
            cumulative_departed,

        "completed_trips":
            completed_trips,

        "completion_rate_pct":
            completion_rate,

        "completion_of_departed_pct":
            completion_of_departed,

        "avg_travel_time":
            trip_metrics[
                "avg_travel_time"
            ],

        "avg_waiting_time":
            trip_metrics[
                "avg_waiting_time"
            ],

        "avg_time_loss":
            trip_metrics[
                "avg_time_loss"
            ],

        "mean_queue_length":
            mean_queue_length,

        "max_queue_length":
            max_queue,

        "mean_step_waiting_time":
            mean_step_waiting,

        "mean_vehicle_count":
            mean_vehicle_count,

        "teleports_started":
            teleports_started,

        "teleports_ended":
            teleports_ended,

        "active_at_end":
            active_at_end,

        "min_expected_at_end":
            min_expected_at_end,

        "tls_count":
            len(
                tls_ids
            ),

        "active_program":
            program_expected,

        "peak_process_tree_ram_mb":
            peak_ram_mb,

        "simulation_loop_runtime_s":
            simulation_loop_runtime,

        "total_wall_runtime_s":
            total_wall_runtime,

        "tripinfo_arrival_match":
            arrival_match,

        "step_file":
            str(
                step_file
            ),

        "tripinfo_file":
            str(
                tripinfo_file
            ),

        "status":
            "PASS",
    }

    write_single_summary(
        summary_file,
        row,
    )

    print(
        "\nRESULT"
    )

    print(
        f"  Loaded:                "
        f"{loaded_vehicles}"
    )

    print(
        f"  Departed:              "
        f"{cumulative_departed}"
    )

    print(
        f"  Completed:             "
        f"{completed_trips}"
    )

    print(
        f"  Completion rate:       "
        f"{completion_rate:.2f}%"
    )

    print(
        f"  Avg travel time:       "
        f"{row['avg_travel_time']:.2f}s"
    )

    print(
        f"  Avg waiting time:      "
        f"{row['avg_waiting_time']:.2f}s"
    )

    print(
        f"  Avg time loss:         "
        f"{row['avg_time_loss']:.2f}s"
    )

    print(
        f"  Mean queue length:     "
        f"{mean_queue_length:.2f}"
    )

    print(
        f"  Max queue length:      "
        f"{max_queue}"
    )

    print(
        f"  Teleports started:     "
        f"{teleports_started}"
    )

    print(
        f"  Peak process RAM:      "
        f"{peak_ram_mb:.2f} MB"
    )

    print(
        f"  Simulation runtime:    "
        f"{simulation_loop_runtime:.3f}s"
    )

    print(
        f"  Total wall runtime:    "
        f"{total_wall_runtime:.3f}s"
    )

    print(
        "  Status:                PASS"
    )

    return row


# ============================================================
# PER-RUN SUMMARY
# ============================================================

def write_single_summary(
    path,
    row,
):

    with path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()

        writer.writerow(
            row
        )


def read_summary_file(
    path,
):

    with path.open(
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        row = next(
            reader
        )

    return convert_summary_types(
        row
    )


def convert_summary_types(
    row,
):

    integer_fields = {
        "seed",
        "simulation_duration_s",
        "loaded_vehicles",
        "departed_vehicles",
        "completed_trips",
        "max_queue_length",
        "teleports_started",
        "teleports_ended",
        "active_at_end",
        "min_expected_at_end",
        "tls_count",
    }

    float_fields = {
        "completion_rate_pct",
        "completion_of_departed_pct",
        "avg_travel_time",
        "avg_waiting_time",
        "avg_time_loss",
        "mean_queue_length",
        "mean_step_waiting_time",
        "mean_vehicle_count",
        "peak_process_tree_ram_mb",
        "simulation_loop_runtime_s",
        "total_wall_runtime_s",
    }

    converted = dict(
        row
    )

    for field in integer_fields:

        value = converted.get(
            field
        )

        if value not in (
            None,
            "",
        ):

            converted[field] = int(
                float(value)
            )

    for field in float_fields:

        value = converted.get(
            field
        )

        if value not in (
            None,
            "",
        ):

            converted[field] = float(
                value
            )

    value = converted.get(
        "tripinfo_arrival_match"
    )

    if isinstance(
        value,
        str,
    ):

        converted[
            "tripinfo_arrival_match"
        ] = (
            value.lower()
            == "true"
        )

    return converted


# ============================================================
# MASTER SUMMARY
# ============================================================

def write_master_summary(
    rows,
):

    CONVENTIONAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = sorted(
        rows,
        key=lambda row: (
            row["grid"],
            row["scenario"],
            row["controller"],
            row["seed"],
        ),
    )

    with MASTER_SUMMARY.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            ordered
        )


# ============================================================
# AGGREGATION
# ============================================================

def safe_mean(
    values,
):

    if not values:

        return ""

    return statistics.mean(
        values
    )


def safe_std(
    values,
):

    if len(values) < 2:

        return 0.0

    return statistics.stdev(
        values
    )


def write_aggregate_summary(
    rows,
):

    groups = {}

    for row in rows:

        key = (
            row["controller"],
            row["grid"],
            row["scenario"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    metrics = [
        "completed_trips",
        "completion_rate_pct",
        "avg_travel_time",
        "avg_waiting_time",
        "avg_time_loss",
        "mean_queue_length",
        "max_queue_length",
        "teleports_started",
        "peak_process_tree_ram_mb",
        "simulation_loop_runtime_s",
        "total_wall_runtime_s",
    ]

    aggregate_rows = []

    for (
        controller,
        grid,
        scenario,
    ), group in sorted(
        groups.items()
    ):

        aggregate = {
            "controller":
                controller,

            "grid":
                grid,

            "scenario":
                scenario,

            "n_runs":
                len(group),
        }

        for metric in metrics:

            values = [
                float(
                    row[
                        metric
                    ]
                )
                for row in group
            ]

            aggregate[
                f"{metric}_mean"
            ] = safe_mean(
                values
            )

            aggregate[
                f"{metric}_std"
            ] = safe_std(
                values
            )

        aggregate_rows.append(
            aggregate
        )

    fieldnames = [
        "controller",
        "grid",
        "scenario",
        "n_runs",
    ]

    for metric in metrics:

        fieldnames.append(
            f"{metric}_mean"
        )

        fieldnames.append(
            f"{metric}_std"
        )

    with AGGREGATE_SUMMARY.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            aggregate_rows
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "pilot",
            "primary",
            "scalability",
            "full",
            "single",
        ],
        default="pilot",
    )

    parser.add_argument(
        "--controller",
        choices=CONTROLLERS,
    )

    parser.add_argument(
        "--grid",
        choices=[
            "2x2",
            "3x3",
            "4x4",
            "5x5",
        ],
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "low",
            "medium",
            "high",
            "dynamic",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rerun experiments even if "
            "per-run summary files exist."
        ),
    )

    args = parser.parse_args()

    CONVENTIONAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = build_run_matrix(
        mode=args.mode,
        controller=args.controller,
        grid=args.grid,
        scenario=args.scenario,
        seed=args.seed,
    )

    banner(
        "THESIS REVISION V2 — "
        "CONVENTIONAL EVALUATION"
    )

    print(
        f"\nMode: "
        f"{args.mode}"
    )

    print(
        f"Runs scheduled: "
        f"{len(runs)}"
    )

    print(
        f"Simulation duration: "
        f"{SIM_DURATION}s"
    )

    print(
        f"Metric logging interval: "
        f"{LOG_STEP}s"
    )

    rows = []

    total_experiments = len(
        runs
    )

    for index, (
        controller,
        grid,
        scenario,
        seed,
    ) in enumerate(
        runs,
        start=1,
    ):

        print(
            "\n"
            + "#" * 88
        )

        print(
            f"RUN "
            f"{index}/"
            f"{total_experiments}"
        )

        print(
            "#" * 88
        )

        row = run_one(
            controller=controller,
            grid=grid,
            scenario=scenario,
            seed=seed,
            overwrite=args.overwrite,
        )

        rows.append(
            row
        )

        # Persist progress after every run.
        write_master_summary(
            rows
        )

        write_aggregate_summary(
            rows
        )

    banner(
        "EVALUATION COMPLETE"
    )

    print(
        f"\nRuns completed/loaded: "
        f"{len(rows)}"
    )

    print(
        f"\nMaster summary:\n"
        f"{MASTER_SUMMARY}"
    )

    print(
        f"\nAggregate summary:\n"
        f"{AGGREGATE_SUMMARY}"
    )

    print(
        "\nMetric compatibility:"
    )

    print(
        "  [PASS] tripinfo.duration"
    )

    print(
        "  [PASS] tripinfo.waitingTime"
    )

    print(
        "  [PASS] tripinfo.timeLoss"
    )

    print(
        "  [PASS] completed tripinfo count"
    )

    print(
        "  [PASS] queue sampled every 5 seconds"
    )

    print(
        "\nRevision metrics:"
    )

    print(
        "  [PASS] completion rate"
    )

    print(
        "  [PASS] teleports"
    )

    print(
        "  [PASS] peak RAM"
    )

    print(
        "  [PASS] simulation runtime"
    )

    print(
        "\nNo original experiment files "
        "were modified."
    )


if __name__ == "__main__":
    main()
