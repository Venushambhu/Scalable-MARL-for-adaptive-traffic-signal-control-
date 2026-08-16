"""
rebuild_webster_v2.py

Regenerates ONLY the Webster demand-tuned fixed-time baselines
with explicit timing parameters for Thesis Revision V2.

Explicit Webster parameters:
    yellow time   = 3 s
    all-red time  = 0 s
    lost time     = 4 s
    minimum green = 10 s
    minimum cycle = 20 s
    maximum cycle = 120 s

The previous generated Webster directory is backed up.
Original network and route files are never modified.
"""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from common import (
    NETWORK_FILES,
    ROUTE_FILES,
    BASELINES_V2_DIR,
    GRID_AGENT_COUNTS,
)


# ============================================================
# CONFIGURATION
# ============================================================

YELLOW_TIME = 3
ALL_RED_TIME = 0
LOST_TIME = 4
MIN_GREEN = 10
MIN_CYCLE = 20
MAX_CYCLE = 120

WEBSTER_DIR = (
    BASELINES_V2_DIR
    / "tuned_fixed"
)

BACKUP_DIR = (
    BASELINES_V2_DIR
    / "tuned_fixed_default_yellow4_backup"
)

EXPERIMENTS = [
    ("2x2", "low"),
    ("2x2", "medium"),
    ("2x2", "high"),
    ("2x2", "dynamic"),
    ("3x3", "medium"),
    ("4x4", "medium"),
    ("5x5", "medium"),
]


# ============================================================
# HELPERS
# ============================================================

def banner(text):

    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def get_tool():

    sumo_home = os.environ.get(
        "SUMO_HOME"
    )

    if not sumo_home:

        raise RuntimeError(
            "SUMO_HOME is not defined."
        )

    tool = (
        Path(sumo_home)
        / "tools"
        / "tlsCycleAdaptation.py"
    )

    if not tool.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{tool}"
        )

    return tool


def run_command(command):

    print(
        "\nCOMMAND:"
    )

    print(
        " ".join(
            str(value)
            for value in command
        )
    )

    result = subprocess.run(
        [
            str(value)
            for value in command
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.stdout.strip():

        print(
            "\nSUMO OUTPUT:"
        )

        print(
            result.stdout.strip()
        )

    if result.returncode != 0:

        raise RuntimeError(
            f"Webster generation failed "
            f"with exit code "
            f"{result.returncode}."
        )


def get_tllogics(path):

    tree = ET.parse(
        path
    )

    root = tree.getroot()

    return root.findall(
        ".//tlLogic"
    )


def classify_state(state):

    green = any(
        c in state
        for c in ("G", "g")
    )

    yellow = any(
        c in state
        for c in ("Y", "y")
    )

    if green and yellow:
        return "MIXED"

    if green:
        return "GREEN"

    if yellow:
        return "YELLOW"

    return "OTHER"


# ============================================================
# BACKUP OLD GENERATED WEBSTER FILES
# ============================================================

def backup_previous_outputs():

    banner(
        "A. BACKING UP PREVIOUS WEBSTER OUTPUTS"
    )

    if not WEBSTER_DIR.exists():

        print(
            "No previous Webster directory exists."
        )

        return

    if BACKUP_DIR.exists():

        print(
            f"Backup already exists:\n"
            f"{BACKUP_DIR}"
        )

        print(
            "Existing backup will be preserved."
        )

    else:

        shutil.copytree(
            WEBSTER_DIR,
            BACKUP_DIR,
        )

        print(
            f"PASS: previous Webster files "
            f"backed up to:\n"
            f"{BACKUP_DIR}"
        )

    shutil.rmtree(
        WEBSTER_DIR
    )

    WEBSTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "PASS: clean tuned_fixed directory created."
    )


# ============================================================
# BUILD ONE WEBSTER PLAN
# ============================================================

def build_one(
    grid,
    scenario,
    tool,
):

    network_file = (
        NETWORK_FILES[
            grid
        ]
    )

    route_file = (
        ROUTE_FILES[
            (
                grid,
                scenario,
            )
        ]
    )

    output_dir = (
        WEBSTER_DIR
        / f"grid{grid}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / (
            f"grid{grid}_"
            f"{scenario}_"
            f"webster.add.xml"
        )
    )

    banner(
        f"WEBSTER: {grid} / {scenario}"
    )

    command = [
        sys.executable,
        tool,

        "-n",
        network_file,

        "-r",
        route_file,

        "-o",
        output_file,

        "-b",
        "0",

        "-y",
        str(YELLOW_TIME),

        "-a",
        str(ALL_RED_TIME),

        "-l",
        str(LOST_TIME),

        "-g",
        str(MIN_GREEN),

        "--min-cycle",
        str(MIN_CYCLE),

        "--max-cycle",
        str(MAX_CYCLE),
    ]

    run_command(
        command
    )

    if not output_file.exists():

        raise RuntimeError(
            "Expected Webster output "
            f"was not created:\n"
            f"{output_file}"
        )

    return output_file


# ============================================================
# AUDIT ONE GENERATED PLAN
# ============================================================

def audit_one(
    grid,
    scenario,
    path,
):

    logics = get_tllogics(
        path
    )

    expected_tls = (
        GRID_AGENT_COUNTS[
            grid
        ]
    )

    unique_ids = {
        logic.attrib.get(
            "id"
        )
        for logic in logics
    }

    if len(unique_ids) != expected_tls:

        raise AssertionError(
            f"{grid}/{scenario}: "
            f"expected {expected_tls} TLS, "
            f"found {len(unique_ids)}."
        )

    cycles = []

    all_yellows = []

    minimum_green_observed = None

    for logic in logics:

        phases = logic.findall(
            "phase"
        )

        cycle = sum(
            float(
                phase.attrib.get(
                    "duration",
                    0,
                )
            )
            for phase in phases
        )

        cycles.append(
            cycle
        )

        for phase in phases:

            duration = float(
                phase.attrib.get(
                    "duration",
                    0,
                )
            )

            phase_type = (
                classify_state(
                    phase.attrib.get(
                        "state",
                        "",
                    )
                )
            )

            if phase_type == "YELLOW":

                all_yellows.append(
                    duration
                )

            if phase_type == "GREEN":

                if (
                    minimum_green_observed
                    is None
                    or duration
                    < minimum_green_observed
                ):

                    minimum_green_observed = (
                        duration
                    )

    if not all_yellows:

        raise AssertionError(
            f"{grid}/{scenario}: "
            "no yellow phases found."
        )

    bad_yellow = [
        value
        for value in all_yellows
        if abs(
            value
            - YELLOW_TIME
        ) > 1e-9
    ]

    if bad_yellow:

        raise AssertionError(
            f"{grid}/{scenario}: "
            f"unexpected yellow values "
            f"{bad_yellow}"
        )

    if (
        minimum_green_observed
        is not None
        and minimum_green_observed
        < MIN_GREEN
    ):

        raise AssertionError(
            f"{grid}/{scenario}: "
            f"green below {MIN_GREEN}s: "
            f"{minimum_green_observed}s"
        )

    print(
        f"\nPASS: {grid}/{scenario}"
    )

    print(
        f"  TLS count: "
        f"{len(unique_ids)}"
    )

    print(
        f"  Cycle range: "
        f"{min(cycles):.1f}"
        f"–"
        f"{max(cycles):.1f}s"
    )

    print(
        f"  Minimum observed green: "
        f"{minimum_green_observed:.1f}s"
    )

    print(
        f"  Yellow phases: "
        f"{YELLOW_TIME}s"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "THESIS REVISION V2 — "
        "REBUILD WEBSTER BASELINES"
    )

    print(
        "\nExplicit parameters:"
    )

    print(
        f"  Yellow time:    "
        f"{YELLOW_TIME}s"
    )

    print(
        f"  All-red time:   "
        f"{ALL_RED_TIME}s"
    )

    print(
        f"  Lost time:      "
        f"{LOST_TIME}s"
    )

    print(
        f"  Minimum green:  "
        f"{MIN_GREEN}s"
    )

    print(
        f"  Minimum cycle:  "
        f"{MIN_CYCLE}s"
    )

    print(
        f"  Maximum cycle:  "
        f"{MAX_CYCLE}s"
    )

    tool = get_tool()

    print(
        f"\nTool:\n{tool}"
    )

    backup_previous_outputs()

    banner(
        "B. GENERATING CORRECTED WEBSTER PLANS"
    )

    generated = []

    for grid, scenario in EXPERIMENTS:

        output = build_one(
            grid,
            scenario,
            tool,
        )

        generated.append(
            (
                grid,
                scenario,
                output,
            )
        )

    banner(
        "C. AUDITING CORRECTED WEBSTER PLANS"
    )

    for (
        grid,
        scenario,
        output,
    ) in generated:

        audit_one(
            grid,
            scenario,
            output,
        )

    banner(
        "PASS: CORRECTED WEBSTER "
        "BASELINES GENERATED"
    )

    print(
        "\nAll Webster plans now use:"
    )

    print(
        "  [PASS] 3-second yellow"
    )

    print(
        "  [PASS] 0-second all-red"
    )

    print(
        "  [PASS] 10-second minimum green constraint"
    )

    print(
        "  [PASS] explicit Webster lost time = 4 s"
    )

    print(
        "  [PASS] cycle search range = 20–120 s"
    )

    print(
        "\nOriginal networks and routes "
        "were not modified."
    )


if __name__ == "__main__":
    main()
