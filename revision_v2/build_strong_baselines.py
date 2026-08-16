"""
build_strong_baselines.py

Builds stronger conventional traffic-signal baselines for the
revised thesis experiments.

Baselines
---------
1. Original fixed-time
   Existing network:
       42 s green
        3 s yellow
       42 s green
        3 s yellow

2. Webster demand-tuned fixed-time
   Generated with SUMO's official tlsCycleAdaptation.py.

3. SUMO actuated control
   Generated with netconvert using:
       --tls.rebuild
       --tls.default-type actuated

Original thesis files are NEVER overwritten.

Outputs
-------
baselines_v2/
    tuned_fixed/
    actuated/
    baseline_build_manifest.csv
"""

from pathlib import Path
import csv
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from common import (
    PROJECT_ROOT,
    NETWORK_FILES,
    ROUTE_FILES,
    BASELINES_V2_DIR,
    GRID_AGENT_COUNTS,
)


# ============================================================
# CONFIGURATION
# ============================================================

TUNED_DIR = (
    BASELINES_V2_DIR
    / "tuned_fixed"
)

ACTUATED_DIR = (
    BASELINES_V2_DIR
    / "actuated"
)

MANIFEST_FILE = (
    BASELINES_V2_DIR
    / "baseline_build_manifest.csv"
)


# Match MARL safety/control assumptions where possible.
ACTUATED_MIN_GREEN = 10

# SUMO default max green for generated actuated TLS is 50 s.
ACTUATED_MAX_GREEN = 50

YELLOW_TIME = 3

ALL_RED_TIME = 0


# All experimentally used network/scenario combinations.
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
# UTILITIES
# ============================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def find_tools():

    sumo_home = os.environ.get(
        "SUMO_HOME"
    )

    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME is not defined."
        )

    sumo_home = Path(
        sumo_home
    )

    tls_adaptation = (
        sumo_home
        / "tools"
        / "tlsCycleAdaptation.py"
    )

    netconvert = shutil.which(
        "netconvert"
    )

    if not tls_adaptation.exists():
        raise FileNotFoundError(
            f"tlsCycleAdaptation.py not found:\n"
            f"{tls_adaptation}"
        )

    if not netconvert:
        raise FileNotFoundError(
            "netconvert was not found in PATH."
        )

    return (
        tls_adaptation,
        Path(netconvert),
    )


def run_command(command, description):

    print("\n" + "-" * 78)
    print(description)
    print("-" * 78)

    print(
        "COMMAND:\n"
        + " ".join(
            str(x)
            for x in command
        )
    )

    result = subprocess.run(
        [
            str(x)
            for x in command
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
            f"\nCommand failed with exit code "
            f"{result.returncode}:\n"
            f"{description}"
        )

    return result


# ============================================================
# ROUTE VALIDATION
# ============================================================

def inspect_route_file(route_file):
    """
    tlsCycleAdaptation.py expects vehicles with nested
    <route> definitions rather than flow/trip-only input.

    This check prevents us from blindly running Webster
    adaptation against unsupported route structures.
    """

    tree = ET.parse(
        route_file
    )

    root = tree.getroot()

    vehicles = root.findall(
        ".//vehicle"
    )

    flows = root.findall(
        ".//flow"
    )

    trips = root.findall(
        ".//trip"
    )

    vehicles_with_route_child = 0

    vehicles_with_route_attribute = 0

    for vehicle in vehicles:

        if vehicle.find("route") is not None:
            vehicles_with_route_child += 1

        if vehicle.get("route") is not None:
            vehicles_with_route_attribute += 1

    return {
        "vehicles": len(vehicles),
        "flows": len(flows),
        "trips": len(trips),
        "vehicles_with_route_child":
            vehicles_with_route_child,
        "vehicles_with_route_attribute":
            vehicles_with_route_attribute,
    }


def validate_webster_route(
    grid,
    scenario,
    route_file,
):
    stats = inspect_route_file(
        route_file
    )

    print(
        f"\nRoute audit: {grid} / {scenario}"
    )

    print(
        f"  Vehicles: "
        f"{stats['vehicles']}"
    )

    print(
        f"  Vehicle <route> children: "
        f"{stats['vehicles_with_route_child']}"
    )

    print(
        f"  Vehicle route attributes: "
        f"{stats['vehicles_with_route_attribute']}"
    )

    print(
        f"  Flows: "
        f"{stats['flows']}"
    )

    print(
        f"  Trips: "
        f"{stats['trips']}"
    )

    # Strongest directly supported format for
    # tlsCycleAdaptation.py:
    #
    # <vehicle ...>
    #     <route edges="..."/>
    # </vehicle>

    supported = (
        stats[
            "vehicles_with_route_child"
        ]
        > 0
        and stats["flows"] == 0
        and stats["trips"] == 0
    )

    return (
        supported,
        stats,
    )


# ============================================================
# ACTUATED NETWORK GENERATION
# ============================================================

def build_actuated_network(
    grid,
    network_file,
    netconvert,
):
    """
    Build an actuated copy of one network.

    Each network only needs to be converted once because
    the same actuated TLS algorithm can then be evaluated
    against different route scenarios.
    """

    output_dir = (
        ACTUATED_DIR
        / f"grid{grid}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"grid{grid}_actuated.net.xml"
    )

    command = [
        netconvert,

        "-s",
        network_file,

        "-o",
        output_file,

        "--tls.rebuild",

        "--tls.default-type",
        "actuated",

        "--tls.min-dur",
        str(
            ACTUATED_MIN_GREEN
        ),

        "--tls.max-dur",
        str(
            ACTUATED_MAX_GREEN
        ),

        "--tls.yellow.time",
        str(
            YELLOW_TIME
        ),

        "--tls.allred.time",
        str(
            ALL_RED_TIME
        ),
    ]

    run_command(
        command,
        (
            f"Building actuated network "
            f"for grid {grid}"
        ),
    )

    if not output_file.exists():
        raise RuntimeError(
            f"Expected actuated network "
            f"was not generated:\n"
            f"{output_file}"
        )

    print(
        f"\nPASS: actuated network created:\n"
        f"{output_file}"
    )

    return output_file


# ============================================================
# WEBSTER FIXED-TIME GENERATION
# ============================================================

def build_tuned_fixed(
    grid,
    scenario,
    network_file,
    route_file,
    tls_adaptation,
):
    """
    Generate demand-informed fixed-time TLS definitions
    using SUMO's tlsCycleAdaptation.py.

    The output is an additional .add.xml file. It is NOT
    a replacement network file.

    SUMO later loads it with:

        -a generated_file.add.xml
    """

    output_dir = (
        TUNED_DIR
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

    supported, stats = (
        validate_webster_route(
            grid,
            scenario,
            route_file,
        )
    )

    if not supported:

        print(
            "\nSKIP: route structure is not "
            "directly supported by "
            "tlsCycleAdaptation.py."
        )

        return (
            None,
            stats,
            "SKIPPED_UNSUPPORTED_ROUTE_FORMAT",
        )

    command = [
        sys.executable,
        tls_adaptation,

        "-n",
        network_file,

        "-r",
        route_file,

        "-o",
        output_file,

        "-b",
        "0",
    ]

    run_command(
        command,
        (
            "Building Webster demand-tuned "
            f"fixed-time plan: "
            f"{grid} / {scenario}"
        ),
    )

    if not output_file.exists():
        raise RuntimeError(
            f"Expected Webster output "
            f"was not generated:\n"
            f"{output_file}"
        )

    print(
        f"\nPASS: Webster plan created:\n"
        f"{output_file}"
    )

    return (
        output_file,
        stats,
        "CREATED",
    )


# ============================================================
# BASIC OUTPUT AUDIT
# ============================================================

def audit_actuated_network(
    network_file,
    expected_agents,
):
    tree = ET.parse(
        network_file
    )

    root = tree.getroot()

    logics = root.findall(
        "tlLogic"
    )

    unique_ids = {
        logic.get("id")
        for logic in logics
    }

    types = sorted(
        {
            logic.get(
                "type",
                "unknown",
            )
            for logic in logics
        }
    )

    print(
        f"\nActuated TLS audit:"
    )

    print(
        f"  TLS IDs: "
        f"{len(unique_ids)}"
    )

    print(
        f"  Expected: "
        f"{expected_agents}"
    )

    print(
        f"  TLS types: "
        f"{types}"
    )

    if (
        len(unique_ids)
        != expected_agents
    ):
        raise AssertionError(
            "Actuated network TLS count "
            "does not match expected "
            "agent count."
        )

    if "actuated" not in types:
        raise AssertionError(
            "Generated network does not "
            "contain actuated tlLogic."
        )

    print(
        "PASS: actuated TLS count/type "
        "validation succeeded."
    )


def audit_webster_file(
    additional_file,
):
    tree = ET.parse(
        additional_file
    )

    root = tree.getroot()

    logics = root.findall(
        ".//tlLogic"
    )

    if not logics:
        raise AssertionError(
            f"No tlLogic elements found in:\n"
            f"{additional_file}"
        )

    print(
        f"\nWebster output audit:"
    )

    print(
        f"  TLS programs: "
        f"{len(logics)}"
    )

    cycle_lengths = []

    for logic in logics:

        phases = logic.findall(
            "phase"
        )

        cycle = sum(
            float(
                phase.get(
                    "duration",
                    0,
                )
            )
            for phase in phases
        )

        cycle_lengths.append(
            cycle
        )

    if cycle_lengths:

        print(
            f"  Minimum cycle: "
            f"{min(cycle_lengths):.1f}s"
        )

        print(
            f"  Maximum cycle: "
            f"{max(cycle_lengths):.1f}s"
        )

    print(
        "PASS: Webster output contains "
        "traffic-light programs."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "THESIS REVISION V2 — "
        "BUILD STRONG BASELINES"
    )

    tls_adaptation, netconvert = (
        find_tools()
    )

    print(
        f"\nPython: "
        f"{sys.executable}"
    )

    print(
        f"tlsCycleAdaptation.py:\n"
        f"{tls_adaptation}"
    )

    print(
        f"\nnetconvert:\n"
        f"{netconvert}"
    )

    TUNED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ACTUATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    # --------------------------------------------------------
    # A. BUILD ACTUATED NETWORKS
    # --------------------------------------------------------

    banner(
        "A. BUILDING ACTUATED NETWORKS"
    )

    actuated_outputs = {}

    for grid in [
        "2x2",
        "3x3",
        "4x4",
        "5x5",
    ]:

        output = build_actuated_network(
            grid=grid,
            network_file=NETWORK_FILES[grid],
            netconvert=netconvert,
        )

        audit_actuated_network(
            network_file=output,
            expected_agents=(
                GRID_AGENT_COUNTS[
                    grid
                ]
            ),
        )

        actuated_outputs[
            grid
        ] = output

        manifest_rows.append(
            {
                "controller":
                    "actuated",
                "grid":
                    grid,
                "scenario":
                    "all_applicable",
                "status":
                    "CREATED",
                "output_file":
                    str(output),
                "route_vehicles":
                    "",
                "route_flows":
                    "",
                "route_trips":
                    "",
            }
        )

    # --------------------------------------------------------
    # B. WEBSTER TUNED FIXED-TIME
    # --------------------------------------------------------

    banner(
        "B. BUILDING WEBSTER "
        "DEMAND-TUNED FIXED-TIME PLANS"
    )

    for grid, scenario in EXPERIMENTS:

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

        (
            output,
            stats,
            status,
        ) = build_tuned_fixed(
            grid=grid,
            scenario=scenario,
            network_file=network_file,
            route_file=route_file,
            tls_adaptation=(
                tls_adaptation
            ),
        )

        if output is not None:

            audit_webster_file(
                output
            )

        manifest_rows.append(
            {
                "controller":
                    "webster_fixed",
                "grid":
                    grid,
                "scenario":
                    scenario,
                "status":
                    status,
                "output_file":
                    (
                        str(output)
                        if output
                        else ""
                    ),
                "route_vehicles":
                    stats[
                        "vehicles"
                    ],
                "route_flows":
                    stats[
                        "flows"
                    ],
                "route_trips":
                    stats[
                        "trips"
                    ],
            }
        )

    # --------------------------------------------------------
    # C. WRITE MANIFEST
    # --------------------------------------------------------

    banner(
        "C. WRITING BUILD MANIFEST"
    )

    fields = [
        "controller",
        "grid",
        "scenario",
        "status",
        "output_file",
        "route_vehicles",
        "route_flows",
        "route_trips",
    ]

    with MANIFEST_FILE.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    print(
        f"\nManifest written to:\n"
        f"{MANIFEST_FILE}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    banner(
        "BASELINE BUILD COMPLETE"
    )

    print(
        "\nOriginal thesis files "
        "were NOT modified."
    )

    print(
        "\nActuated networks:"
    )

    for grid, path in (
        actuated_outputs.items()
    ):

        print(
            f"  {grid}: {path}"
        )

    print(
        "\nWebster plans are stored under:"
    )

    print(
        f"  {TUNED_DIR}"
    )

    print(
        "\nNext step:"
    )

    print(
        "  audit every generated TLS "
        "program before evaluation."
    )


if __name__ == "__main__":
    main()
