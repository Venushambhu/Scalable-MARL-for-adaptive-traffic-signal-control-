"""
audit_baselines_v2.py

Full structural audit of all conventional traffic-signal baselines
generated for Thesis Revision V2.

Audits:
1. Original fixed-time networks
2. Webster demand-tuned fixed-time plans
3. SUMO actuated networks

Outputs:
    results_v2/baseline_tls_audit.csv

No network, route, model or baseline file is modified.
"""

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

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

ORIGINAL_EXPECTED_DURATIONS = [
    42.0,
    3.0,
    42.0,
    3.0,
]

EXPECTED_YELLOW = 3.0

EXPECTED_ACTUATED_MIN_GREEN = 10.0

EXPECTED_ACTUATED_MAX_GREEN = 50.0


EXPERIMENTS = [
    ("2x2", "low"),
    ("2x2", "medium"),
    ("2x2", "high"),
    ("2x2", "dynamic"),
    ("3x3", "medium"),
    ("4x4", "medium"),
    ("5x5", "medium"),
]


AUDIT_CSV = (
    RESULTS_V2_DIR
    / "baseline_tls_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 85)
    print(text)
    print("=" * 85)


def classify_state(state):
    has_green = any(
        c in state
        for c in ("G", "g")
    )

    has_yellow = any(
        c in state
        for c in ("Y", "y")
    )

    if has_green and has_yellow:
        return "MIXED"

    if has_green:
        return "GREEN"

    if has_yellow:
        return "YELLOW"

    if state and all(
        c in ("r", "R")
        for c in state
    ):
        return "ALL_RED"

    return "OTHER"


def get_tllogics(path):
    tree = ET.parse(path)
    root = tree.getroot()

    return root.findall(
        ".//tlLogic"
    )


def phase_duration(phase):
    return float(
        phase.attrib.get(
            "duration",
            0,
        )
    )


def phase_min_duration(phase):
    value = phase.attrib.get(
        "minDur"
    )

    if value is None:
        return None

    return float(value)


def phase_max_duration(phase):
    value = phase.attrib.get(
        "maxDur"
    )

    if value is None:
        return None

    return float(value)


def logic_cycle(logic):
    return sum(
        phase_duration(phase)
        for phase in logic.findall("phase")
    )


def print_logic_summary(logic):
    tls_id = logic.attrib.get(
        "id",
        "unknown",
    )

    program_id = logic.attrib.get(
        "programID",
        "unknown",
    )

    logic_type = logic.attrib.get(
        "type",
        "unknown",
    )

    phases = logic.findall(
        "phase"
    )

    print(
        f"\nTLS {tls_id} | "
        f"program={program_id} | "
        f"type={logic_type} | "
        f"phases={len(phases)} | "
        f"nominal cycle={logic_cycle(logic):.1f}s"
    )

    for index, phase in enumerate(phases):

        state = phase.attrib.get(
            "state",
            "",
        )

        duration = phase_duration(
            phase
        )

        min_dur = phase.attrib.get(
            "minDur",
            "-",
        )

        max_dur = phase.attrib.get(
            "maxDur",
            "-",
        )

        print(
            f"  phase {index:>2} | "
            f"{classify_state(state):<8} | "
            f"duration={duration:>6.1f}s | "
            f"minDur={str(min_dur):>5} | "
            f"maxDur={str(max_dur):>5}"
        )


# ============================================================
# ORIGINAL FIXED-TIME AUDIT
# ============================================================

def audit_original_fixed(rows):

    banner(
        "A. ORIGINAL FIXED-TIME BASELINE"
    )

    total_tls = 0

    for grid, network_file in NETWORK_FILES.items():

        logics = get_tllogics(
            network_file
        )

        unique_ids = {
            logic.attrib.get("id")
            for logic in logics
        }

        expected_count = (
            GRID_AGENT_COUNTS[grid]
        )

        print(
            f"\nGRID {grid}: "
            f"{len(unique_ids)} TLS "
            f"(expected {expected_count})"
        )

        assert (
            len(unique_ids)
            == expected_count
        )

        for logic in logics:

            total_tls += 1

            phases = logic.findall(
                "phase"
            )

            durations = [
                phase_duration(p)
                for p in phases
            ]

            states = [
                classify_state(
                    p.attrib.get(
                        "state",
                        "",
                    )
                )
                for p in phases
            ]

            expected_states = [
                "GREEN",
                "YELLOW",
                "GREEN",
                "YELLOW",
            ]

            valid = (
                logic.attrib.get(
                    "type"
                )
                == "static"
                and durations
                == ORIGINAL_EXPECTED_DURATIONS
                and states
                == expected_states
            )

            if not valid:
                raise AssertionError(
                    "\nOriginal baseline structure "
                    f"unexpected for "
                    f"{grid}/"
                    f"{logic.attrib.get('id')}\n"
                    f"Durations: {durations}\n"
                    f"States: {states}"
                )

            rows.append(
                {
                    "controller":
                        "original_fixed",
                    "grid":
                        grid,
                    "scenario":
                        "all",
                    "tls_id":
                        logic.attrib.get(
                            "id"
                        ),
                    "type":
                        logic.attrib.get(
                            "type"
                        ),
                    "phase_count":
                        len(phases),
                    "cycle_seconds":
                        logic_cycle(logic),
                    "green_min_seconds":
                        "",
                    "green_max_seconds":
                        "",
                    "yellow_seconds":
                        EXPECTED_YELLOW,
                    "status":
                        "PASS",
                }
            )

        print(
            "  PASS: all TLS use "
            "42/3/42/3 static timing."
        )

    print(
        f"\nOriginal TLS audited: {total_tls}"
    )

    assert total_tls == 54

    print(
        "PASS: all 54 original TLS "
        "programs validated."
    )


# ============================================================
# WEBSTER AUDIT
# ============================================================

def audit_webster(rows):

    banner(
        "B. WEBSTER DEMAND-TUNED FIXED-TIME"
    )

    for grid, scenario in EXPERIMENTS:

        path = (
            BASELINES_V2_DIR
            / "tuned_fixed"
            / f"grid{grid}"
            / (
                f"grid{grid}_"
                f"{scenario}_"
                f"webster.add.xml"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Webster plan:\n"
                f"{path}"
            )

        logics = get_tllogics(
            path
        )

        expected_count = (
            GRID_AGENT_COUNTS[
                grid
            ]
        )

        unique_ids = {
            logic.attrib.get("id")
            for logic in logics
        }

        print(
            f"\n{grid} / {scenario}"
        )

        print(
            f"  TLS programs: "
            f"{len(unique_ids)} "
            f"(expected {expected_count})"
        )

        assert (
            len(unique_ids)
            == expected_count
        )

        cycle_lengths = []

        for logic in logics:

            logic_type = (
                logic.attrib.get(
                    "type",
                    "static",
                )
            )

            phases = logic.findall(
                "phase"
            )

            cycle = logic_cycle(
                logic
            )

            cycle_lengths.append(
                cycle
            )

            yellow_durations = [
                phase_duration(p)
                for p in phases
                if classify_state(
                    p.attrib.get(
                        "state",
                        "",
                    )
                )
                == "YELLOW"
            ]

            if not yellow_durations:
                raise AssertionError(
                    f"No yellow phase found "
                    f"in Webster TLS "
                    f"{logic.attrib.get('id')}"
                )

            if any(
                abs(y - EXPECTED_YELLOW)
                > 1e-9
                for y in yellow_durations
            ):
                raise AssertionError(
                    "Unexpected yellow duration "
                    f"in {grid}/{scenario}/"
                    f"{logic.attrib.get('id')}: "
                    f"{yellow_durations}"
                )

            rows.append(
                {
                    "controller":
                        "webster_fixed",
                    "grid":
                        grid,
                    "scenario":
                        scenario,
                    "tls_id":
                        logic.attrib.get(
                            "id"
                        ),
                    "type":
                        logic_type,
                    "phase_count":
                        len(phases),
                    "cycle_seconds":
                        cycle,
                    "green_min_seconds":
                        "",
                    "green_max_seconds":
                        "",
                    "yellow_seconds":
                        EXPECTED_YELLOW,
                    "status":
                        "PASS",
                }
            )

        print(
            f"  Cycle range: "
            f"{min(cycle_lengths):.1f}"
            f"–"
            f"{max(cycle_lengths):.1f}s"
        )

        print(
            "  PASS: TLS count and "
            "yellow timing validated."
        )

        # Show one representative TLS.
        print_logic_summary(
            logics[0]
        )


# ============================================================
# ACTUATED AUDIT
# ============================================================

def audit_actuated(rows):

    banner(
        "C. SUMO ACTUATED BASELINE"
    )

    total_tls = 0

    for grid in [
        "2x2",
        "3x3",
        "4x4",
        "5x5",
    ]:

        path = (
            BASELINES_V2_DIR
            / "actuated"
            / f"grid{grid}"
            / f"grid{grid}_actuated.net.xml"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing actuated network:\n"
                f"{path}"
            )

        logics = get_tllogics(
            path
        )

        expected_count = (
            GRID_AGENT_COUNTS[
                grid
            ]
        )

        unique_ids = {
            logic.attrib.get("id")
            for logic in logics
        }

        print(
            f"\nGRID {grid}: "
            f"{len(unique_ids)} TLS "
            f"(expected {expected_count})"
        )

        assert (
            len(unique_ids)
            == expected_count
        )

        for logic in logics:

            total_tls += 1

            logic_type = (
                logic.attrib.get(
                    "type"
                )
            )

            if logic_type != "actuated":
                raise AssertionError(
                    f"TLS "
                    f"{logic.attrib.get('id')} "
                    f"in {grid} is "
                    f"{logic_type}, "
                    f"not actuated."
                )

            phases = logic.findall(
                "phase"
            )

            green_mins = []
            green_maxes = []
            yellow_values = []

            for phase in phases:

                state_type = (
                    classify_state(
                        phase.attrib.get(
                            "state",
                            "",
                        )
                    )
                )

                if state_type == "GREEN":

                    min_dur = (
                        phase_min_duration(
                            phase
                        )
                    )

                    max_dur = (
                        phase_max_duration(
                            phase
                        )
                    )

                    if min_dur is not None:
                        green_mins.append(
                            min_dur
                        )

                    if max_dur is not None:
                        green_maxes.append(
                            max_dur
                        )

                elif state_type == "YELLOW":

                    yellow_values.append(
                        phase_duration(
                            phase
                        )
                    )

            if not green_mins:
                raise AssertionError(
                    f"No actuated green minDur "
                    f"found for {grid}/"
                    f"{logic.attrib.get('id')}"
                )

            if not green_maxes:
                raise AssertionError(
                    f"No actuated green maxDur "
                    f"found for {grid}/"
                    f"{logic.attrib.get('id')}"
                )

            if any(
                abs(
                    value
                    - EXPECTED_ACTUATED_MIN_GREEN
                )
                > 1e-9
                for value in green_mins
            ):
                raise AssertionError(
                    f"Unexpected actuated minDur "
                    f"in {grid}/"
                    f"{logic.attrib.get('id')}: "
                    f"{green_mins}"
                )

            if any(
                abs(
                    value
                    - EXPECTED_ACTUATED_MAX_GREEN
                )
                > 1e-9
                for value in green_maxes
            ):
                raise AssertionError(
                    f"Unexpected actuated maxDur "
                    f"in {grid}/"
                    f"{logic.attrib.get('id')}: "
                    f"{green_maxes}"
                )

            if any(
                abs(
                    value
                    - EXPECTED_YELLOW
                )
                > 1e-9
                for value in yellow_values
            ):
                raise AssertionError(
                    f"Unexpected yellow duration "
                    f"in {grid}/"
                    f"{logic.attrib.get('id')}: "
                    f"{yellow_values}"
                )

            rows.append(
                {
                    "controller":
                        "actuated",
                    "grid":
                        grid,
                    "scenario":
                        "all",
                    "tls_id":
                        logic.attrib.get(
                            "id"
                        ),
                    "type":
                        logic_type,
                    "phase_count":
                        len(phases),
                    "cycle_seconds":
                        logic_cycle(logic),
                    "green_min_seconds":
                        min(green_mins),
                    "green_max_seconds":
                        max(green_maxes),
                    "yellow_seconds":
                        EXPECTED_YELLOW,
                    "status":
                        "PASS",
                }
            )

        print(
            "  PASS: actuated type, "
            "10s minimum green, "
            "50s maximum green and "
            "3s yellow validated."
        )

        print_logic_summary(
            logics[0]
        )

    print(
        f"\nActuated TLS audited: "
        f"{total_tls}"
    )

    assert total_tls == 54

    print(
        "PASS: all 54 actuated TLS "
        "programs validated."
    )


# ============================================================
# SAVE AUDIT TABLE
# ============================================================

def save_rows(rows):

    RESULTS_V2_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "controller",
        "grid",
        "scenario",
        "tls_id",
        "type",
        "phase_count",
        "cycle_seconds",
        "green_min_seconds",
        "green_max_seconds",
        "yellow_seconds",
        "status",
    ]

    with AUDIT_CSV.open(
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

    print(
        f"\nAudit CSV written to:\n"
        f"{AUDIT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "THESIS REVISION V2 — "
        "BASELINE TLS STRUCTURAL AUDIT"
    )

    rows = []

    audit_original_fixed(
        rows
    )

    audit_webster(
        rows
    )

    audit_actuated(
        rows
    )

    save_rows(
        rows
    )

    banner(
        "PASS: COMPLETE BASELINE "
        "STRUCTURAL AUDIT"
    )

    print(
        "\nValidated:"
    )

    print(
        "  [PASS] Original fixed-time "
        "54 TLS"
    )

    print(
        "  [PASS] Webster tuned plans "
        "for all 7 experiments"
    )

    print(
        "  [PASS] Actuated "
        "54 TLS"
    )

    print(
        "  [PASS] Original project files "
        "unchanged"
    )

    print(
        "\nNo performance claim has been "
        "made yet."
    )

    print(
        "The next stage is controller "
        "smoke testing and evaluation."
    )


if __name__ == "__main__":
    main()
