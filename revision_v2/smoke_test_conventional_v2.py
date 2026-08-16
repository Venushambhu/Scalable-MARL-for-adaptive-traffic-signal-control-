"""
smoke_test_conventional_v2.py

Operational smoke test for the three conventional controllers used in
Thesis Revision V2.

Controllers
-----------
1. Original fixed-time
2. Webster demand-tuned fixed-time
3. SUMO actuated

Scenario
--------
Grid:       2x2
Demand:     medium
Seed:       11
Duration:   1800 s

Purpose
-------
This is NOT the final statistical evaluation.

It validates that:
- every controller starts correctly,
- the intended TLS program is actually active,
- vehicles load and arrive,
- queues can be measured,
- teleports are recorded,
- TLS phase behaviour is observable.

Output
------
results_v2/conventional_smoke_test.csv

No original files are modified.
"""

import csv
import shutil
import time

import traci

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    BASELINES_V2_DIR,
    RESULTS_V2_DIR,
    GRID_AGENT_COUNTS,
)


# ============================================================
# CONFIGURATION
# ============================================================

GRID = "2x2"
SCENARIO = "medium"

SEED = 11
SIM_DURATION = 1800

EXPECTED_TLS = GRID_AGENT_COUNTS[GRID]

OUTPUT_CSV = (
    RESULTS_V2_DIR
    / "conventional_smoke_test.csv"
)


# ============================================================
# CONTROLLER PATHS
# ============================================================

ORIGINAL_NETWORK = (
    NETWORK_FILES[GRID]
)

ROUTE_FILE = (
    ROUTE_FILES[
        (
            GRID,
            SCENARIO,
        )
    ]
)

WEBSTER_FILE = (
    BASELINES_V2_DIR
    / "tuned_fixed"
    / f"grid{GRID}"
    / (
        f"grid{GRID}_"
        f"{SCENARIO}_"
        f"webster.add.xml"
    )
)

ACTUATED_NETWORK = (
    BASELINES_V2_DIR
    / "actuated"
    / f"grid{GRID}"
    / f"grid{GRID}_actuated.net.xml"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):

    print("\n" + "=" * 82)
    print(text)
    print("=" * 82)


def validate_files():

    required = [
        ORIGINAL_NETWORK,
        ROUTE_FILE,
        WEBSTER_FILE,
        ACTUATED_NETWORK,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing:\n"
                f"{path}"
            )

    print(
        "PASS: all controller/network/"
        "route files exist."
    )


def build_sumo_command(
    network_file,
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
        str(ROUTE_FILE),

        "--seed",
        str(SEED),

        "--no-warnings",
        "true",

        "--no-step-log",
        "true",

        "--time-to-teleport",
        "300",
    ]

    if additional_file is not None:

        command += [
            "-a",
            str(additional_file),
        ]

    return command


# ============================================================
# PHASE TRACE
# ============================================================

def initialise_phase_trace(
    representative_tls,
):

    phase = (
        traci.trafficlight
        .getPhase(
            representative_tls
        )
    )

    return {
        "last_phase": phase,
        "segment_start": 0,
        "segments": [],
    }


def update_phase_trace(
    trace,
    representative_tls,
    simulation_time,
):

    phase = (
        traci.trafficlight
        .getPhase(
            representative_tls
        )
    )

    if phase != trace["last_phase"]:

        duration = (
            simulation_time
            - trace["segment_start"]
        )

        trace["segments"].append(
            (
                trace["last_phase"],
                duration,
            )
        )

        trace["last_phase"] = phase

        trace["segment_start"] = (
            simulation_time
        )


def finish_phase_trace(
    trace,
    simulation_time,
):

    duration = (
        simulation_time
        - trace["segment_start"]
    )

    trace["segments"].append(
        (
            trace["last_phase"],
            duration,
        )
    )


def get_green_durations(
    trace,
):

    # For these networks:
    #
    # phase 0 = green
    # phase 1 = yellow
    # phase 2 = green
    # phase 3 = yellow

    return [
        duration
        for phase, duration
        in trace["segments"]
        if phase in (
            0,
            2,
        )
    ]


# ============================================================
# RUN ONE CONTROLLER
# ============================================================

def run_controller(
    controller_name,
    network_file,
    additional_file=None,
    required_program=None,
):

    banner(
        f"RUNNING: {controller_name}"
    )

    command = build_sumo_command(
        network_file=network_file,
        additional_file=additional_file,
    )

    print(
        f"\nNetwork:\n"
        f"{network_file}"
    )

    print(
        f"\nRoute:\n"
        f"{ROUTE_FILE}"
    )

    if additional_file is not None:

        print(
            f"\nAdditional TLS file:\n"
            f"{additional_file}"
        )

    wall_start = time.perf_counter()

    traci.start(
        command
    )

    try:

        tls_ids = sorted(
            traci.trafficlight
            .getIDList()
        )

        print(
            f"\nTraffic lights detected: "
            f"{len(tls_ids)}"
        )

        if len(tls_ids) != EXPECTED_TLS:

            raise AssertionError(
                f"Expected {EXPECTED_TLS} TLS, "
                f"found {len(tls_ids)}."
            )

        representative_tls = (
            tls_ids[0]
        )

        print(
            f"Representative TLS: "
            f"{representative_tls}"
        )

        # ----------------------------------------------------
        # PROGRAM ACTIVATION
        # ----------------------------------------------------

        if required_program is not None:

            print(
                f"\nActivating TLS program "
                f"'{required_program}'..."
            )

            for tls in tls_ids:

                available_programs = {
                    logic.programID
                    for logic in (
                        traci.trafficlight
                        .getAllProgramLogics(
                            tls
                        )
                    )
                }

                if (
                    required_program
                    not in available_programs
                ):

                    raise AssertionError(
                        f"TLS {tls} does not contain "
                        f"required program "
                        f"'{required_program}'. "
                        f"Available programs: "
                        f"{sorted(available_programs)}"
                    )

                traci.trafficlight.setProgram(
                    tls,
                    required_program,
                )

        # Standardise initial phase.
        for tls in tls_ids:

            traci.trafficlight.setPhase(
                tls,
                0,
            )

        active_programs = {
            tls:
                traci.trafficlight
                .getProgram(tls)
            for tls in tls_ids
        }

        print(
            "\nActive programs:"
        )

        for tls, program in (
            active_programs.items()
        ):

            print(
                f"  {tls}: {program}"
            )

        if required_program is not None:

            bad_programs = {
                tls: program
                for tls, program
                in active_programs.items()
                if program
                != required_program
            }

            if bad_programs:

                raise AssertionError(
                    f"Required program "
                    f"'{required_program}' "
                    f"is not active on all TLS:\n"
                    f"{bad_programs}"
                )

        print(
            "PASS: intended TLS program "
            "is active."
        )

        # ----------------------------------------------------
        # LANES FOR QUEUE MEASUREMENT
        # ----------------------------------------------------

        lane_ids = [
            lane
            for lane in (
                traci.lane.getIDList()
            )
            if not lane.startswith(":")
        ]

        print(
            f"Non-internal lanes monitored: "
            f"{len(lane_ids)}"
        )

        # ----------------------------------------------------
        # METRIC ACCUMULATORS
        # ----------------------------------------------------

        cumulative_departed = 0
        cumulative_arrived = 0
        cumulative_teleports = 0

        queue_sum = 0.0
        max_queue = 0

        observed_seconds = 0

        phase_trace = (
            initialise_phase_trace(
                representative_tls
            )
        )

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        while (
            traci.simulation.getTime()
            < SIM_DURATION
        ):

            traci.simulationStep()

            simulation_time = int(
                traci.simulation.getTime()
            )

            departed = (
                traci.simulation
                .getDepartedNumber()
            )

            arrived = (
                traci.simulation
                .getArrivedNumber()
            )

            teleports = (
                traci.simulation
                .getStartingTeleportNumber()
            )

            cumulative_departed += (
                departed
            )

            cumulative_arrived += (
                arrived
            )

            cumulative_teleports += (
                teleports
            )

            network_queue = sum(
                traci.lane
                .getLastStepHaltingNumber(
                    lane
                )
                for lane in lane_ids
            )

            queue_sum += (
                network_queue
            )

            max_queue = max(
                max_queue,
                network_queue,
            )

            observed_seconds += 1

            update_phase_trace(
                trace=phase_trace,
                representative_tls=(
                    representative_tls
                ),
                simulation_time=(
                    simulation_time
                ),
            )

        final_time = int(
            traci.simulation.getTime()
        )

        finish_phase_trace(
            trace=phase_trace,
            simulation_time=final_time,
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        mean_queue = (
            queue_sum
            / observed_seconds
            if observed_seconds
            else 0.0
        )

        completion_rate = (
            cumulative_arrived
            / cumulative_departed
            * 100.0
            if cumulative_departed
            else 0.0
        )

        active_at_end = (
            traci.vehicle.getIDCount()
        )

        min_expected_at_end = (
            traci.simulation
            .getMinExpectedNumber()
        )

        green_durations = (
            get_green_durations(
                phase_trace
            )
        )

        wall_runtime = (
            time.perf_counter()
            - wall_start
        )

        print(
            "\nRESULTS"
        )

        print(
            f"  Simulation time:      "
            f"{final_time}s"
        )

        print(
            f"  Departed vehicles:    "
            f"{cumulative_departed}"
        )

        print(
            f"  Arrived vehicles:     "
            f"{cumulative_arrived}"
        )

        print(
            f"  Completion rate:      "
            f"{completion_rate:.2f}%"
        )

        print(
            f"  Active at end:        "
            f"{active_at_end}"
        )

        print(
            f"  Min expected at end:  "
            f"{min_expected_at_end}"
        )

        print(
            f"  Starting teleports:   "
            f"{cumulative_teleports}"
        )

        print(
            f"  Mean network queue:   "
            f"{mean_queue:.2f}"
        )

        print(
            f"  Maximum network queue:"
            f" {max_queue}"
        )

        print(
            f"  Wall runtime:         "
            f"{wall_runtime:.3f}s"
        )

        print(
            f"\nRepresentative TLS "
            f"{representative_tls} "
            f"green durations:"
        )

        if green_durations:

            print(
                "  "
                + ", ".join(
                    f"{value:.0f}s"
                    for value
                    in green_durations[:20]
                )
            )

            if len(green_durations) > 20:

                print(
                    f"  ... "
                    f"({len(green_durations)} "
                    f"green segments total)"
                )

        else:

            print(
                "  No complete green "
                "segments observed."
            )

        print(
            "\nPASS: controller completed "
            "the smoke-test simulation."
        )

        return {
            "controller":
                controller_name,
            "grid":
                GRID,
            "scenario":
                SCENARIO,
            "seed":
                SEED,
            "simulation_duration_s":
                final_time,
            "departed":
                cumulative_departed,
            "arrived":
                cumulative_arrived,
            "completion_rate_percent":
                completion_rate,
            "active_at_end":
                active_at_end,
            "min_expected_at_end":
                min_expected_at_end,
            "starting_teleports":
                cumulative_teleports,
            "mean_network_queue":
                mean_queue,
            "max_network_queue":
                max_queue,
            "representative_tls":
                representative_tls,
            "green_segments_observed":
                len(green_durations),
            "green_min_observed_s":
                (
                    min(green_durations)
                    if green_durations
                    else ""
                ),
            "green_max_observed_s":
                (
                    max(green_durations)
                    if green_durations
                    else ""
                ),
            "wall_runtime_s":
                wall_runtime,
            "status":
                "PASS",
        }

    finally:

        traci.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    rows,
):

    RESULTS_V2_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "controller",
        "grid",
        "scenario",
        "seed",
        "simulation_duration_s",
        "departed",
        "arrived",
        "completion_rate_percent",
        "active_at_end",
        "min_expected_at_end",
        "starting_teleports",
        "mean_network_queue",
        "max_network_queue",
        "representative_tls",
        "green_segments_observed",
        "green_min_observed_s",
        "green_max_observed_s",
        "wall_runtime_s",
        "status",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"\nSmoke-test CSV:\n"
        f"{OUTPUT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "THESIS REVISION V2 — "
        "CONVENTIONAL CONTROLLER SMOKE TEST"
    )

    print(
        f"\nGrid:       {GRID}"
    )

    print(
        f"Scenario:   {SCENARIO}"
    )

    print(
        f"Seed:       {SEED}"
    )

    print(
        f"Duration:   {SIM_DURATION}s"
    )

    validate_files()

    rows = []

    # --------------------------------------------------------
    # 1. ORIGINAL FIXED-TIME
    # --------------------------------------------------------

    rows.append(
        run_controller(
            controller_name=(
                "original_fixed"
            ),
            network_file=(
                ORIGINAL_NETWORK
            ),
        )
    )

    # --------------------------------------------------------
    # 2. WEBSTER FIXED-TIME
    #
    # CRITICAL:
    # Loading the .add.xml alone is not enough for this
    # validation. Explicitly activate program "a".
    # --------------------------------------------------------

    rows.append(
        run_controller(
            controller_name=(
                "webster_fixed"
            ),
            network_file=(
                ORIGINAL_NETWORK
            ),
            additional_file=(
                WEBSTER_FILE
            ),
            required_program="a",
        )
    )

    # --------------------------------------------------------
    # 3. SUMO ACTUATED
    # --------------------------------------------------------

    rows.append(
        run_controller(
            controller_name=(
                "actuated"
            ),
            network_file=(
                ACTUATED_NETWORK
            ),
        )
    )

    save_results(
        rows
    )

    banner(
        "PASS: ALL CONVENTIONAL "
        "CONTROLLER SMOKE TESTS COMPLETED"
    )

    print(
        "\nControllers validated:"
    )

    print(
        "  [PASS] Original fixed-time"
    )

    print(
        "  [PASS] Webster tuned fixed-time"
    )

    print(
        "  [PASS] SUMO actuated"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are diagnostic smoke-test "
        "results, not final thesis statistics."
    )


if __name__ == "__main__":
    main()
