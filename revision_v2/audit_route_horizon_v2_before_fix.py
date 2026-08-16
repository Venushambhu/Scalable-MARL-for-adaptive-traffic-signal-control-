"""
audit_route_horizon_v2.py

Checks whether every experimental route file is compatible with
the 1800-second simulation horizon used in Thesis Revision V2.

For each route file the script reports:
    - total vehicle definitions
    - earliest scheduled departure
    - latest scheduled departure
    - vehicles scheduled by the simulation horizon
    - vehicles scheduled after the simulation horizon
    - percentage of demand visible within the horizon

No files are modified.
"""

import csv
import xml.etree.ElementTree as ET

from common import (
    ROUTE_FILES,
    RESULTS_V2_DIR,
    SIM_DURATION,
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


OUTPUT_CSV = (
    RESULTS_V2_DIR
    / "route_horizon_audit.csv"
)


def banner(text):
    print("\n" + "=" * 82)
    print(text)
    print("=" * 82)


def parse_time(value):
    """
    Parse SUMO numeric seconds or HH:MM:SS-style time.
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


def audit_route(
    grid,
    scenario,
    route_file,
):
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

        depart_raw = vehicle.get(
            "depart"
        )

        depart = parse_time(
            depart_raw
        )

        if depart is None:

            non_numeric_departures.append(
                (
                    vehicle.get("id"),
                    depart_raw,
                )
            )

        else:

            numeric_departures.append(
                depart
            )

    if not numeric_departures:

        raise RuntimeError(
            f"No numeric vehicle departure "
            f"times found in:\n{route_file}"
        )

    earliest = min(
        numeric_departures
    )

    latest = max(
        numeric_departures
    )

    scheduled_by_horizon = sum(
        1
        for depart in numeric_departures
        if depart <= SIM_DURATION
    )

    scheduled_after_horizon = sum(
        1
        for depart in numeric_departures
        if depart > SIM_DURATION
    )

    coverage_percent = (
        scheduled_by_horizon
        / len(numeric_departures)
        * 100.0
    )

    status = (
        "PASS"
        if scheduled_after_horizon == 0
        else "HORIZON_MISMATCH"
    )

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
        f"  Earliest departure: "
        f"{earliest:.1f}s"
    )

    print(
        f"  Latest departure:   "
        f"{latest:.1f}s"
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
            f"  WARNING: "
            f"{len(non_numeric_departures)} "
            f"non-numeric departure definitions."
        )

    return {
        "grid": grid,
        "scenario": scenario,
        "route_file": str(route_file),
        "total_vehicles": len(vehicles),
        "numeric_departures":
            len(numeric_departures),
        "non_numeric_departures":
            len(non_numeric_departures),
        "earliest_departure_s":
            earliest,
        "latest_departure_s":
            latest,
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

    mismatches = []

    for grid, scenario in EXPERIMENTS:

        route_file = ROUTE_FILES[
            (
                grid,
                scenario,
            )
        ]

        row = audit_route(
            grid,
            scenario,
            route_file,
        )

        rows.append(
            row
        )

        if (
            row["status"]
            == "HORIZON_MISMATCH"
        ):

            mismatches.append(
                row
            )

    RESULTS_V2_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
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
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    banner(
        "ROUTE HORIZON AUDIT SUMMARY"
    )

    print(
        f"\nAudit CSV:\n"
        f"{OUTPUT_CSV}"
    )

    if not mismatches:

        print(
            "\nPASS: all experimental "
            "vehicles are scheduled within "
            f"the {SIM_DURATION}s horizon."
        )

        print(
            "\nWe can proceed directly to "
            "conventional-controller smoke testing."
        )

    else:

        print(
            f"\nIMPORTANT: "
            f"{len(mismatches)} experiment(s) "
            "contain vehicles scheduled after "
            f"{SIM_DURATION}s."
        )

        for row in mismatches:

            print(
                f"\n  {row['grid']} / "
                f"{row['scenario']}"
            )

            print(
                f"    latest departure = "
                f"{row['latest_departure_s']:.1f}s"
            )

            print(
                f"    vehicles after horizon = "
                f"{row['scheduled_after_horizon']}"
            )

            print(
                f"    visible demand = "
                f"{row['coverage_percent']:.2f}%"
            )

        print(
            "\nSTOP: do not start new training "
            "or final evaluation until the "
            "simulation-horizon design is resolved."
        )


if __name__ == "__main__":
    main()
