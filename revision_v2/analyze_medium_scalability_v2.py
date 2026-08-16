"""
Authoritative medium-demand scalability analysis.

Grids:
    2x2 = 4 agents
    3x3 = 9 agents
    4x4 = 16 agents
    5x5 = 25 agents

Controllers:
    original_fixed
    webster_fixed
    actuated
    dqn_v2
    ppo_v2

Runs:
    4 grids x 5 controllers x 5 seeds
    = 100 final evaluation runs
"""

import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results_v2"

OUTPUT_DIR = (
    RESULTS
    / "analysis"
    / "medium_scalability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GRIDS = [
    "2x2",
    "3x3",
    "4x4",
    "5x5",
]

AGENTS = {
    "2x2": 4,
    "3x3": 9,
    "4x4": 16,
    "5x5": 25,
}

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


# ============================================================
# CONVENTIONAL DATA
# ============================================================

baseline_path = (
    RESULTS
    / "conventional"
    / "conventional_all_runs_summary.csv"
)

baseline_all = read_csv(
    baseline_path
)

baseline_rows = [
    row
    for row in baseline_all
    if (
        row["grid"] in GRIDS
        and row["scenario"] == "medium"
        and row["controller"] in [
            "original_fixed",
            "webster_fixed",
            "actuated",
        ]
    )
]

assert len(baseline_rows) == 60, (
    f"Expected 60 conventional medium rows, "
    f"found {len(baseline_rows)}"
)


combined = []


for row in baseline_rows:

    normalized = dict(row)

    normalized[
        "controller_final"
    ] = row["controller"]

    normalized[
        "algorithm_family"
    ] = "conventional"

    normalized[
        "agent_count"
    ] = AGENTS[
        row["grid"]
    ]

    combined.append(
        normalized
    )


# ============================================================
# LEARNED DATA
# ============================================================

learned_count = 0

for grid in GRIDS:

    for algorithm in [
        "dqn",
        "ppo",
    ]:

        path = (
            RESULTS
            / "learned"
            / f"grid{grid}"
            / algorithm
            / "medium"
            / f"{algorithm}_medium_all_seeds.csv"
        )

        rows = read_csv(
            path
        )

        assert len(rows) == 5, (
            f"{path}: expected 5 rows, "
            f"found {len(rows)}"
        )

        seeds = sorted(
            int(row["seed"])
            for row in rows
        )

        assert seeds == [
            11,
            12,
            13,
            14,
            15,
        ]

        assert all(
            row.get("status") == "PASS"
            for row in rows
        )

        for row in rows:

            normalized = dict(row)

            normalized["grid"] = grid
            normalized["scenario"] = "medium"

            normalized[
                "controller_final"
            ] = (
                f"{algorithm}_v2"
            )

            normalized[
                "algorithm_family"
            ] = "learned"

            normalized[
                "agent_count"
            ] = AGENTS[
                grid
            ]

            combined.append(
                normalized
            )

            learned_count += 1


assert learned_count == 40

assert len(combined) == 100, (
    f"Expected 100 total scalability rows, "
    f"found {len(combined)}"
)


# ============================================================
# VERIFY EVERY GRID / CONTROLLER CELL
# ============================================================

for grid in GRIDS:

    for controller in CONTROLLERS:

        rows = [
            row
            for row in combined
            if (
                row["grid"] == grid
                and
                row["controller_final"]
                == controller
            )
        ]

        assert len(rows) == 5, (
            f"{grid}/{controller}: "
            f"expected 5 rows, found {len(rows)}"
        )

        seeds = sorted(
            int(row["seed"])
            for row in rows
        )

        assert seeds == [
            11,
            12,
            13,
            14,
            15,
        ]


# ============================================================
# SAVE RAW 100-RUN DATASET
# ============================================================

raw_output = (
    OUTPUT_DIR
    / "combined_medium_scalability_100_runs.csv"
)

fields = []

for row in combined:

    for key in row.keys():

        if key not in fields:
            fields.append(key)


with raw_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        combined
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for grid in GRIDS:

    for controller in CONTROLLERS:

        rows = [
            row
            for row in combined
            if (
                row["grid"] == grid
                and
                row["controller_final"]
                == controller
            )
        ]

        summary = {
            "grid":
                grid,

            "agent_count":
                AGENTS[grid],

            "controller":
                controller,

            "n":
                5,
        }

        for metric in METRICS:

            values = [
                float(
                    row[metric]
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
            ] = statistics.stdev(
                values
            )

        summary_rows.append(
            summary
        )


summary_output = (
    OUTPUT_DIR
    / "summary_medium_scalability.csv"
)

with summary_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# PRINT TABLES
# ============================================================

for grid in GRIDS:

    print()
    print("=" * 112)

    print(
        f"{grid} MEDIUM | "
        f"{AGENTS[grid]} AGENTS | "
        "FIVE-SEED SUMMARY"
    )

    print("=" * 112)

    print(
        f"{'Controller':<18}"
        f"{'Comp.%':>10}"
        f"{'Travel':>12}"
        f"{'Waiting':>12}"
        f"{'TimeLoss':>12}"
        f"{'Queue':>11}"
        f"{'Tele':>9}"
    )

    print("-" * 112)

    rows = [
        row
        for row in summary_rows
        if row["grid"] == grid
    ]

    for controller in CONTROLLERS:

        row = next(
            r
            for r in rows
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
# DESCRIPTIVE BEST CONTROLLER
# ============================================================

print()
print("=" * 112)
print(
    "DESCRIPTIVE BEST CONTROLLER BY GRID"
)
print(
    "(means only — no significance claim)"
)
print("=" * 112)


for grid in GRIDS:

    rows = [
        row
        for row in summary_rows
        if row["grid"] == grid
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
        f"{grid} "
        f"({AGENTS[grid]} agents)"
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
        f"{best_queue['mean_queue_length_mean']:.3f}"
        f")"
    )


# ============================================================
# DQN VS PPO SCALING
# ============================================================

print()
print("=" * 112)
print("DQN VS PPO ACROSS AGENT COUNTS")
print("=" * 112)

print(
    f"{'Agents':>8}"
    f"{'Grid':>8}"
    f"{'DQN Travel':>14}"
    f"{'PPO Travel':>14}"
    f"{'DQN Wait':>14}"
    f"{'PPO Wait':>14}"
    f"{'DQN Queue':>14}"
    f"{'PPO Queue':>14}"
)

print("-" * 112)


for grid in GRIDS:

    dqn = next(
        row
        for row in summary_rows
        if (
            row["grid"] == grid
            and row["controller"]
            == "dqn_v2"
        )
    )

    ppo = next(
        row
        for row in summary_rows
        if (
            row["grid"] == grid
            and row["controller"]
            == "ppo_v2"
        )
    )

    print(
        f"{AGENTS[grid]:>8}"
        f"{grid:>8}"
        f"{dqn['avg_travel_time_mean']:>14.2f}"
        f"{ppo['avg_travel_time_mean']:>14.2f}"
        f"{dqn['avg_waiting_time_mean']:>14.2f}"
        f"{ppo['avg_waiting_time_mean']:>14.2f}"
        f"{dqn['mean_queue_length_mean']:>14.3f}"
        f"{ppo['mean_queue_length_mean']:>14.3f}"
    )


print()
print("=" * 112)
print("DATASET VALIDATION")
print("=" * 112)

print(
    "Conventional medium runs:",
    len(baseline_rows),
)

print(
    "Learned medium runs:     ",
    learned_count,
)

print(
    "Total scalability runs:  ",
    len(combined),
)

print()
print(
    "Raw dataset:"
)
print(raw_output)

print()
print(
    "Summary:"
)
print(summary_output)

print()
print(
    "PASS: authoritative 100-run "
    "medium scalability dataset created."
)
