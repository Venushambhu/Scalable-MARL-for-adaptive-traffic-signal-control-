"""
audit_route_horizon_v2.py

Audit whether all vehicles in the thesis route files are scheduled
within the 1800-second simulation horizon.

This script DOES NOT modify any network or route files.

Output:
    results_v2/route_horizon_audit.csv
"""

import csv
import xml.etree.ElementTree as ET

from common import (
    ROUTE_FILES,
    RESULTS_V2_DIR,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SIM_DURATION = 1800

EXPERIMENTS = [
    ("2x2", "low"),
    ("2x2", "medium"),
    ("2x2", "high"),
    ("2x2", "dynamic"),
    ("3x3", "medium"),
    ("4x4", "medium"),
    ("5x5", "medium"),
]

OUTPUT_CSV = (
    RESULTS_V2_DIR
    / "route_horizon_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 82)
    print(text)
    print("=" * 82)


def parse_time(value):
    """
    Convert a SUMO departure time to seconds.

    Supports:
        123
        123.5
        HH:MM:SS
        MM:SS
    """

    if value is None:
        return None

    value = str(value).strip()

    try:
        return float(value)

    except ValueError:
        pass

    if ":" in value:

        parts = value.split(":")

        try:
            numbers = [
                float(part)
                for part in parts
            ]

        except ValueError:
            return None

        if len(numbers) == 3:

            hours, minutes, seconds = numbers

            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

        if len(numbers) == 2:

            minutes, seconds = numbers

            return (
                minutes * 60
                + seconds
            )

    return None


# ============================================================
# ROUTE AUDIT
# ============================================================

def audit_route(
    grid,
    scenario,
    route_file,
):
    """
    Inspect all explicit <vehicle> departure times.
    """

    tree = ET.parse(
        route_file
    )

    root = tree.getroot()

    vehicles = root.findall(
        ".//vehicle"
    )

    numeric_departures = []
    non_numeric_departures = []

    for vehicle in vehicles:

        vehicle_id = vehicle.get(
            "id"
        )

        depart_raw = vehicle.get(
            "depart"
        )

        depart_seconds = parse_time(
            depart_raw
        )

        if depart_seconds is None:

            non_numeric_departures.append(
                (
                    vehicle_id,
                    depart_raw,
                )
            )

        else:

            numeric_departures.append(
                depart_seconds
            )

    if not vehicles:

        raise RuntimeError(
            f"No <vehicle> elements found in:\n"
            f"{route_file}"
        )

    if not numeric_departures:

        raise RuntimeError(
            f"No numeric departure times found in:\n"
            f"{route_file}"
        )

    earliest_departure = min(
        numeric_departures
    )

    latest_departure = max(
        numeric_departures
    )

    scheduled_by_horizon = sum(
        1
        for departure
        in numeric_departures
        if departure <= SIM_DURATION
    )

    scheduled_after_horizon = sum(
        1
        for departure
        in numeric_departures
        if departure > SIM_DURATION
    )

    numeric_vehicle_count = len(
        numeric_departures
    )

    coverage_percent = (
        scheduled_by_horizon
        / numeric_vehicle_count
        * 100.0
    )

    if (
        scheduled_after_horizon == 0
        and len(non_numeric_departures) == 0
    ):
        status = "PASS"

    elif scheduled_after_horizon > 0:
        status = "HORIZON_MISMATCH"

    else:
        status = "NON_NUMERIC_DEPARTURE"

    print(
        f"\n{grid} / {scenario}"
    )

    print(
        f"  Route file: "
        f"{route_file}"
    )

    print(
        f"  Vehicles: "
        f"{len(vehicles)}"
    )

    print(
        f"  Numeric departures: "
        f"{numeric_vehicle_count}"
    )

    print(
        f"  Non-numeric departures: "
        f"{len(non_numeric_departures)}"
    )

    print(
        f"  Earliest departure: "
        f"{earliest_departure:.1f}s"
    )

    print(
        f"  Latest departure:   "
        f"{latest_departure:.1f}s"
    )

    print(
        f"  Simulation horizon: "
        f"{SIM_DURATION}s"
    )

    print(
        f"  Scheduled <= horizon: "
        f"{scheduled_by_horizon}"
    )

    print(
        f"  Scheduled > horizon:  "
        f"{scheduled_after_horizon}"
    )

    print(
        f"  Demand coverage: "
        f"{coverage_percent:.2f}%"
    )

    print(
        f"  STATUS: {status}"
    )

    if non_numeric_departures:

        print(
            "\n  WARNING: sample non-numeric "
            "departure values:"
        )

        for (
            vehicle_id,
            value,
        ) in non_numeric_departures[:5]:

            print(
                f"    {vehicle_id}: {value}"
            )

    return {
        "grid": grid,
        "scenario": scenario,
        "route_file": str(route_file),
        "total_vehicles": len(vehicles),
        "numeric_departures":
            numeric_vehicle_count,
        "non_numeric_departures":
            len(non_numeric_departures),
        "earliest_departure_s":
            earliest_departure,
        "latest_departure_s":
            latest_departure,
        "simulation_horizon_s":
            SIM_DURATION,
        "scheduled_by_horizon":
            scheduled_by_horizon,
        "scheduled_after_horizon":
            scheduled_after_horizon,
        "coverage_percent":
            coverage_percent,
        "status":
            status,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(rows):

    RESULTS_V2_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "grid",
        "scenario",
        "route_file",
        "total_vehicles",
        "numeric_departures",
        "non_numeric_departures",
        "earliest_departure_s",
        "latest_departure_s",
        "simulation_horizon_s",
        "scheduled_by_horizon",
        "scheduled_after_horizon",
        "coverage_percent",
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


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "THESIS REVISION V2 — "
        "ROUTE HORIZON AUDIT"
    )

    print(
        f"\nConfigured simulation horizon: "
        f"{SIM_DURATION}s"
    )

    rows = []

    for grid, scenario in EXPERIMENTS:

        key = (
            grid,
            scenario,
        )

        if key not in ROUTE_FILES:

            raise KeyError(
                f"ROUTE_FILES does not contain "
                f"{key}"
            )

        route_file = ROUTE_FILES[
            key
        ]

        row = audit_route(
            grid=grid,
            scenario=scenario,
            route_file=route_file,
        )

        rows.append(
            row
        )

    save_results(
        rows
    )

    mismatches = [
        row
        for row in rows
        if row["status"]
        != "PASS"
    ]

    banner(
        "ROUTE HORIZON AUDIT SUMMARY"
    )

    print(
        f"\nAudit CSV written to:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nExperiments audited: "
        f"{len(rows)}"
    )

    print(
        f"Experiments passed: "
        f"{len(rows) - len(mismatches)}"
    )

    print(
        f"Experiments requiring attention: "
        f"{len(mismatches)}"
    )

    if not mismatches:

        print(
            "\nPASS: ALL ROUTE FILES FIT "
            "WITHIN THE SIMULATION HORIZON"
        )

        print(
            f"\nAll explicitly scheduled vehicles "
            f"depart at or before "
            f"{SIM_DURATION}s."
        )

        print(
            "\nWe can proceed to conventional "
            "controller smoke testing."
        )

    else:

        print(
            "\nATTENTION: ROUTE-HORIZON "
            "ISSUES DETECTED"
        )

        for row in mismatches:

            print(
                f"\n  {row['grid']} / "
                f"{row['scenario']}"
            )

            print(
                f"    status: "
                f"{row['status']}"
            )

            print(
                f"    latest departure: "
                f"{row['latest_departure_s']:.1f}s"
            )

            print(
                f"    vehicles after horizon: "
                f"{row['scheduled_after_horizon']}"
            )

            print(
                f"    coverage: "
                f"{row['coverage_percent']:.2f}%"
            )

        print(
            "\nSTOP: resolve the route-horizon "
            "issue before retraining or final "
            "evaluation."
        )


if __name__ == "__main__":
    main()
