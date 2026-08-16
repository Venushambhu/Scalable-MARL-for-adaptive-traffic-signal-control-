"""
Final 2x2 demand-sensitivity analysis.

Combines:
    60 conventional-controller runs
    40 learned-controller runs

Total:
    100 final evaluation runs

Scenarios:
    low
    medium
    high
    dynamic

Controllers:
    original_fixed
    webster_fixed
    actuated
    dqn_v2
    ppo_v2
"""

import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS = (
    PROJECT_ROOT
    / "results_v2"
)

ANALYSIS_DIR = (
    RESULTS
    / "analysis"
    / "2x2_demand"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BASELINE_CSV = (
    RESULTS
    / "conventional"
    / "conventional_all_runs_summary.csv"
)


SCENARIOS = [
    "low",
    "medium",
    "high",
    "dynamic",
]


CONTROLLERS = [
    "original_fixed",
    "webster_fixed",
    "actuated",
    "dqn_v2",
    "ppo_v2",
]


METRICS = [
    "completion_rate_pct",
    "avg_travel_time",
    "avg_waiting_time",
    "avg_time_loss",
    "mean_queue_length",
    "teleports_started",
]


def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with path.open(
        newline="",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def to_float(row, key):

    return float(
        row[key]
    )


def sample_std(values):

    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


# ============================================================
# LOAD CONVENTIONAL DATA
# ============================================================

baseline_rows_all = read_csv(
    BASELINE_CSV
)

baseline_rows = [
    row
    for row in baseline_rows_all
    if (
        row["grid"] == "2x2"
        and row["scenario"] in SCENARIOS
        and row["controller"] in [
            "original_fixed",
            "webster_fixed",
            "actuated",
        ]
    )
]

assert len(baseline_rows) == 60, (
    f"Expected 60 conventional 2x2 rows, "
    f"found {len(baseline_rows)}"
)


# ============================================================
# NORMALIZE CONVENTIONAL ROWS
# ============================================================

combined = []

for row in baseline_rows:

    normalized = dict(row)

    normalized["controller_final"] = (
        row["controller"]
    )

    normalized["algorithm_family"] = (
        "conventional"
    )

    combined.append(
        normalized
    )


# ============================================================
# LOAD LEARNED DATA
# ============================================================

learned_count = 0

for algorithm in [
    "dqn",
    "ppo",
]:

    for scenario in SCENARIOS:

        path = (
            RESULTS
            / "learned"
            / "grid2x2"
            / algorithm
            / scenario
            / (
                f"{algorithm}_"
                f"{scenario}_"
                f"all_seeds.csv"
            )
        )

        rows = read_csv(
            path
        )

        assert len(rows) == 5, (
            f"{path}: expected 5 rows, "
            f"found {len(rows)}"
        )

        assert all(
            row.get(
                "status",
                "PASS",
            ) == "PASS"
            for row in rows
        )

        for row in rows:

            normalized = dict(row)

            normalized["grid"] = "2x2"

            normalized["scenario"] = (
                scenario
            )

            normalized[
                "controller_final"
            ] = (
                f"{algorithm}_v2"
            )

            normalized[
                "algorithm_family"
            ] = (
                "learned"
            )

            combined.append(
                normalized
            )

            learned_count += 1


assert learned_count == 40

assert len(combined) == 100, (
    f"Expected 100 combined runs, "
    f"found {len(combined)}"
)


# ============================================================
# VERIFY EACH CELL = 5 SEEDS
# ============================================================

for scenario in SCENARIOS:

    for controller in CONTROLLERS:

        rows = [
            row
            for row in combined
            if (
                row["scenario"]
                == scenario
                and
                row["controller_final"]
                == controller
            )
        ]

        assert len(rows) == 5, (
            f"{scenario}/{controller}: "
            f"expected 5 runs, "
            f"found {len(rows)}"
        )


# ============================================================
# WRITE RAW 100-RUN DATASET
# ============================================================

raw_output = (
    ANALYSIS_DIR
    / "combined_2x2_100_runs.csv"
)

all_fields = []

for row in combined:

    for key in row.keys():

        if key not in all_fields:
            all_fields.append(key)


with raw_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=all_fields,
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        combined
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

summary_rows = []

for scenario in SCENARIOS:

    for controller in CONTROLLERS:

        rows = [
            row
            for row in combined
            if (
                row["scenario"]
                == scenario
                and
                row["controller_final"]
                == controller
            )
        ]

        summary = {
            "grid":
                "2x2",

            "scenario":
                scenario,

            "controller":
                controller,

            "n":
                len(rows),
        }

        for metric in METRICS:

            values = [
                to_float(
                    row,
                    metric,
                )
                for row in rows
            ]

            summary[
                f"{metric}_mean"
            ] = statistics.mean(
                values
            )

            summary[
                f"{metric}_std"
            ] = sample_std(
                values
            )

        summary_rows.append(
            summary
        )


summary_output = (
    ANALYSIS_DIR
    / "summary_2x2_demand.csv"
)

with summary_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            summary_rows[0]
            .keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# CONSOLE TABLES
# ============================================================

for scenario in SCENARIOS:

    print()
    print("=" * 110)

    print(
        f"2x2 {scenario.upper()} — "
        "FIVE-CONTROLLER FIVE-SEED SUMMARY"
    )

    print("=" * 110)

    print(
        f"{'Controller':<18}"
        f"{'Comp.%':>10}"
        f"{'Travel':>12}"
        f"{'Waiting':>12}"
        f"{'TimeLoss':>12}"
        f"{'Queue':>11}"
        f"{'Tele':>9}"
    )

    print("-" * 110)

    scenario_summary = [
        row
        for row in summary_rows
        if row["scenario"] == scenario
    ]

    for controller in CONTROLLERS:

        row = next(
            r
            for r in scenario_summary
            if r["controller"]
            == controller
        )

        print(
            f"{controller:<18}"
            f"{row['completion_rate_pct_mean']:>10.2f}"
            f"{row['avg_travel_time_mean']:>12.2f}"
            f"{row['avg_waiting_time_mean']:>12.2f}"
            f"{row['avg_time_loss_mean']:>12.2f}"
            f"{row['mean_queue_length_mean']:>11.2f}"
            f"{row['teleports_started_mean']:>9.2f}"
        )


# ============================================================
# LEARNED VS STRONG BASELINES
# ============================================================

print()
print("=" * 110)
print(
    "DQN/PPO RELATIVE TO ACTUATED AND WEBSTER"
)
print("=" * 110)


def get_summary(
    scenario,
    controller,
):

    return next(
        row
        for row in summary_rows
        if (
            row["scenario"]
            == scenario
            and
            row["controller"]
            == controller
        )
    )


for scenario in SCENARIOS:

    print()
    print(
        f"--- {scenario.upper()} ---"
    )

    for learned in [
        "dqn_v2",
        "ppo_v2",
    ]:

        learner = get_summary(
            scenario,
            learned,
        )

        print()
        print(learned)

        for baseline in [
            "webster_fixed",
            "actuated",
        ]:

            base = get_summary(
                scenario,
                baseline,
            )

            completion_delta = (
                learner[
                    "completion_rate_pct_mean"
                ]
                -
                base[
                    "completion_rate_pct_mean"
                ]
            )

            travel_change = (
                (
                    base[
                        "avg_travel_time_mean"
                    ]
                    -
                    learner[
                        "avg_travel_time_mean"
                    ]
                )
                /
                base[
                    "avg_travel_time_mean"
                ]
                * 100.0
            )

            waiting_change = (
                (
                    base[
                        "avg_waiting_time_mean"
                    ]
                    -
                    learner[
                        "avg_waiting_time_mean"
                    ]
                )
                /
                base[
                    "avg_waiting_time_mean"
                ]
                * 100.0
            )

            queue_change = (
                (
                    base[
                        "mean_queue_length_mean"
                    ]
                    -
                    learner[
                        "mean_queue_length_mean"
                    ]
                )
                /
                base[
                    "mean_queue_length_mean"
                ]
                * 100.0
            )

            print(
                f"  vs {baseline}:"
            )

            print(
                f"    completion delta: "
                f"{completion_delta:+.2f} pp"
            )

            print(
                f"    travel reduction: "
                f"{travel_change:+.2f}%"
            )

            print(
                f"    waiting reduction: "
                f"{waiting_change:+.2f}%"
            )

            print(
                f"    queue reduction: "
                f"{queue_change:+.2f}%"
            )


# ============================================================
# IDENTIFY DESCRIPTIVE BEST
# ============================================================

print()
print("=" * 110)
print(
    "DESCRIPTIVE BEST CONTROLLER BY SCENARIO"
)
print(
    "(No significance claim — descriptive means only)"
)
print("=" * 110)

for scenario in SCENARIOS:

    rows = [
        row
        for row in summary_rows
        if row["scenario"]
        == scenario
    ]

    best_completion = max(
        rows,
        key=lambda r:
            r[
                "completion_rate_pct_mean"
            ],
    )

    best_travel = min(
        rows,
        key=lambda r:
            r[
                "avg_travel_time_mean"
            ],
    )

    best_waiting = min(
        rows,
        key=lambda r:
            r[
                "avg_waiting_time_mean"
            ],
    )

    best_queue = min(
        rows,
        key=lambda r:
            r[
                "mean_queue_length_mean"
            ],
    )

    print()
    print(
        scenario.upper()
    )

    print(
        "  Highest completion: "
        f"{best_completion['controller']} "
        f"("
        f"{best_completion['completion_rate_pct_mean']:.2f}%"
        f")"
    )

    print(
        "  Lowest travel:      "
        f"{best_travel['controller']} "
        f"("
        f"{best_travel['avg_travel_time_mean']:.2f}s"
        f")"
    )

    print(
        "  Lowest waiting:     "
        f"{best_waiting['controller']} "
        f"("
        f"{best_waiting['avg_waiting_time_mean']:.2f}s"
        f")"
    )

    print(
        "  Lowest queue:       "
        f"{best_queue['controller']} "
        f"("
        f"{best_queue['mean_queue_length_mean']:.2f}"
        f")"
    )


print()
print("=" * 110)
print("ANALYSIS DATASET VALIDATION")
print("=" * 110)

print(
    "Conventional runs:",
    len(baseline_rows),
)

print(
    "Learned runs:     ",
    learned_count,
)

print(
    "Total runs:       ",
    len(combined),
)

print()
print(
    "Raw combined dataset:"
)
print(raw_output)

print()
print(
    "Summary dataset:"
)
print(summary_output)

print()
print(
    "PASS: authoritative 100-run "
    "2x2 demand dataset created."
)
