"""
Computational scalability analysis for Thesis Revision V2.

Algorithms:
    DQN
    PPO

Agent counts:
    4  -> 2x2
    9  -> 3x3
    16 -> 4x4
    25 -> 5x5

Uses:
    results_v2/training_compute_master.csv

and deterministic five-seed learned-controller evaluation CSVs.

Measures:
    - training runtime
    - peak training process-tree RAM
    - model size
    - algorithm buffer memory
    - deterministic joint inference latency
    - evaluation process-tree RAM
    - simple scaling ratios
    - descriptive linear-fit R^2

No causal or asymptotic complexity claims are made.
"""

import csv
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results_v2"

TRAINING_CSV = (
    RESULTS
    / "training_compute_master.csv"
)

OUTPUT_DIR = (
    RESULTS
    / "analysis"
    / "computational_scalability"
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

ALGORITHMS = [
    "dqn",
    "ppo",
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


def mean(values):

    return statistics.mean(
        values
    )


def sample_std(values):

    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


def linear_fit(x, y):
    """
    Simple descriptive least-squares fit:
        y = intercept + slope*x

    Returns slope, intercept, R^2.

    This is used only to summarize the observed
    4-point scaling trend.
    """

    x_mean = mean(x)
    y_mean = mean(y)

    ss_x = sum(
        (value - x_mean) ** 2
        for value in x
    )

    if ss_x == 0:
        raise ValueError(
            "Cannot fit constant x."
        )

    slope = (
        sum(
            (xi - x_mean)
            * (yi - y_mean)
            for xi, yi in zip(x, y)
        )
        / ss_x
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    predicted = [
        intercept
        + slope * xi
        for xi in x
    ]

    ss_res = sum(
        (yi - pi) ** 2
        for yi, pi in zip(
            y,
            predicted,
        )
    )

    ss_tot = sum(
        (yi - y_mean) ** 2
        for yi in y
    )

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    return (
        slope,
        intercept,
        r_squared,
    )


# ============================================================
# TRAINING DATA
# ============================================================

training_all = read_csv(
    TRAINING_CSV
)

training = [
    row
    for row in training_all
    if (
        row["scenario"] == "medium"
        and row["grid"] in GRIDS
        and row["algorithm"] in ALGORITHMS
        and row["status"] == "PASS"
        and int(row["total_steps"]) == 50000
    )
]

assert len(training) == 8, (
    f"Expected 8 medium training rows, "
    f"found {len(training)}"
)


# ============================================================
# BUILD COMPUTE SUMMARY
# ============================================================

summary_rows = []


for algorithm in ALGORITHMS:

    for grid in GRIDS:

        train_row = next(
            row
            for row in training
            if (
                row["algorithm"]
                == algorithm
                and row["grid"]
                == grid
            )
        )

        eval_path = (
            RESULTS
            / "learned"
            / f"grid{grid}"
            / algorithm
            / "medium"
            / (
                f"{algorithm}_"
                f"medium_all_seeds.csv"
            )
        )

        eval_rows = read_csv(
            eval_path
        )

        assert len(eval_rows) == 5

        seeds = sorted(
            int(row["seed"])
            for row in eval_rows
        )

        assert seeds == [
            11,
            12,
            13,
            14,
            15,
        ]

        assert all(
            row.get("status")
            == "PASS"
            for row in eval_rows
        )

        joint_latency = [
            float(
                row[
                    "mean_joint_decision_ms"
                ]
            )
            for row in eval_rows
        ]

        p95_joint_latency = [
            float(
                row[
                    "p95_joint_decision_ms"
                ]
            )
            for row in eval_rows
        ]

        eval_ram = [
            float(
                row[
                    "peak_process_tree_ram_mb"
                ]
            )
            for row in eval_rows
        ]

        mean_agent_latency = [
            float(
                row[
                    "mean_agent_predict_ms"
                ]
            )
            for row in eval_rows
        ]

        # Optional wall runtime field.
        wall_values = []

        for row in eval_rows:

            raw = row.get(
                "wall_runtime_s",
                "",
            )

            if raw not in (
                "",
                None,
            ):

                wall_values.append(
                    float(raw)
                )

        training_runtime = float(
            train_row[
                "training_runtime_s"
            ]
        )

        training_ram = float(
            train_row[
                "peak_process_tree_ram_mb"
            ]
        )

        model_mb = float(
            train_row[
                "model_size_total_mb"
            ]
        )

        buffer_mb = float(
            train_row[
                "buffer_total_mb"
            ]
        )

        agents = AGENTS[
            grid
        ]

        summary_rows.append(
            {
                "algorithm":
                    algorithm,

                "grid":
                    grid,

                "agents":
                    agents,

                "training_steps":
                    50000,

                "training_runtime_s":
                    training_runtime,

                "training_runtime_s_per_agent":
                    training_runtime
                    / agents,

                "training_peak_ram_mb":
                    training_ram,

                "training_peak_ram_mb_per_agent":
                    training_ram
                    / agents,

                "buffer_type":
                    train_row[
                        "buffer_type"
                    ],

                "buffer_total_mb":
                    buffer_mb,

                "buffer_mb_per_agent":
                    buffer_mb
                    / agents,

                "model_size_total_mb":
                    model_mb,

                "model_size_mb_per_agent":
                    model_mb
                    / agents,

                "mean_agent_predict_ms":
                    mean(
                        mean_agent_latency
                    ),

                "mean_joint_decision_ms":
                    mean(
                        joint_latency
                    ),

                "std_joint_decision_ms":
                    sample_std(
                        joint_latency
                    ),

                "mean_p95_joint_decision_ms":
                    mean(
                        p95_joint_latency
                    ),

                "decision_budget_pct":
                    (
                        mean(
                            joint_latency
                        )
                        / 5000.0
                        * 100.0
                    ),

                "evaluation_peak_ram_mb_mean":
                    mean(
                        eval_ram
                    ),

                "evaluation_peak_ram_mb_std":
                    sample_std(
                        eval_ram
                    ),

                "evaluation_wall_runtime_s_mean":
                    (
                        mean(
                            wall_values
                        )
                        if wall_values
                        else ""
                    ),
            }
        )


assert len(summary_rows) == 8


# ============================================================
# BASELINE-NORMALIZED SCALING RATIOS
# ============================================================

for algorithm in ALGORITHMS:

    algorithm_rows = [
        row
        for row in summary_rows
        if row["algorithm"]
        == algorithm
    ]

    baseline = next(
        row
        for row in algorithm_rows
        if row["grid"] == "2x2"
    )

    for row in algorithm_rows:

        row[
            "agent_count_ratio_vs_2x2"
        ] = (
            row["agents"]
            / baseline["agents"]
        )

        row[
            "training_runtime_ratio_vs_2x2"
        ] = (
            row["training_runtime_s"]
            / baseline[
                "training_runtime_s"
            ]
        )

        row[
            "training_ram_ratio_vs_2x2"
        ] = (
            row["training_peak_ram_mb"]
            / baseline[
                "training_peak_ram_mb"
            ]
        )

        row[
            "model_size_ratio_vs_2x2"
        ] = (
            row["model_size_total_mb"]
            / baseline[
                "model_size_total_mb"
            ]
        )

        row[
            "joint_latency_ratio_vs_2x2"
        ] = (
            row["mean_joint_decision_ms"]
            / baseline[
                "mean_joint_decision_ms"
            ]
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_csv = (
    OUTPUT_DIR
    / "computational_scalability_summary.csv"
)

with summary_csv.open(
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
# TREND FITS
# ============================================================

trend_rows = []


TREND_METRICS = [
    "training_runtime_s",
    "training_peak_ram_mb",
    "buffer_total_mb",
    "model_size_total_mb",
    "mean_joint_decision_ms",
    "evaluation_peak_ram_mb_mean",
]


for algorithm in ALGORITHMS:

    rows = sorted(
        [
            row
            for row in summary_rows
            if row["algorithm"]
            == algorithm
        ],
        key=lambda row:
            row["agents"],
    )

    x = [
        row["agents"]
        for row in rows
    ]

    for metric in TREND_METRICS:

        y = [
            float(
                row[metric]
            )
            for row in rows
        ]

        (
            slope,
            intercept,
            r_squared,
        ) = linear_fit(
            x,
            y,
        )

        trend_rows.append(
            {
                "algorithm":
                    algorithm,

                "metric":
                    metric,

                "slope_per_additional_agent":
                    slope,

                "intercept":
                    intercept,

                "r_squared":
                    r_squared,
            }
        )


trend_csv = (
    OUTPUT_DIR
    / "computational_scalability_trends.csv"
)

with trend_csv.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            trend_rows[0]
            .keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        trend_rows
    )


# ============================================================
# PRINT MAIN TABLE
# ============================================================

print()
print("=" * 130)
print(
    "DQN/PPO COMPUTATIONAL SCALABILITY"
)
print("=" * 130)

print(
    f"{'Algo':<7}"
    f"{'Grid':<7}"
    f"{'Agents':>7}"
    f"{'Train(s)':>12}"
    f"{'Train RAM':>12}"
    f"{'Buffer MB':>12}"
    f"{'Model MB':>12}"
    f"{'Joint ms':>12}"
    f"{'P95 ms':>12}"
    f"{'Budget %':>12}"
)

print("-" * 130)


for algorithm in ALGORITHMS:

    rows = sorted(
        [
            row
            for row in summary_rows
            if row["algorithm"]
            == algorithm
        ],
        key=lambda row:
            row["agents"],
    )

    for row in rows:

        print(
            f"{algorithm.upper():<7}"
            f"{row['grid']:<7}"
            f"{row['agents']:>7}"
            f"{row['training_runtime_s']:>12.2f}"
            f"{row['training_peak_ram_mb']:>12.2f}"
            f"{row['buffer_total_mb']:>12.3f}"
            f"{row['model_size_total_mb']:>12.3f}"
            f"{row['mean_joint_decision_ms']:>12.4f}"
            f"{row['mean_p95_joint_decision_ms']:>12.4f}"
            f"{row['decision_budget_pct']:>12.5f}"
        )


# ============================================================
# SCALING RATIOS
# ============================================================

print()
print("=" * 130)
print(
    "SCALING RELATIVE TO 2x2 / 4 AGENTS"
)
print("=" * 130)

print(
    f"{'Algo':<7}"
    f"{'Agents':>8}"
    f"{'Agent x':>10}"
    f"{'Runtime x':>12}"
    f"{'RAM x':>10}"
    f"{'Model x':>10}"
    f"{'Latency x':>12}"
)

print("-" * 75)


for algorithm in ALGORITHMS:

    rows = sorted(
        [
            row
            for row in summary_rows
            if row["algorithm"]
            == algorithm
        ],
        key=lambda row:
            row["agents"],
    )

    for row in rows:

        print(
            f"{algorithm.upper():<7}"
            f"{row['agents']:>8}"
            f"{row['agent_count_ratio_vs_2x2']:>10.3f}"
            f"{row['training_runtime_ratio_vs_2x2']:>12.3f}"
            f"{row['training_ram_ratio_vs_2x2']:>10.3f}"
            f"{row['model_size_ratio_vs_2x2']:>10.3f}"
            f"{row['joint_latency_ratio_vs_2x2']:>12.3f}"
        )


# ============================================================
# FIT SUMMARY
# ============================================================

print()
print("=" * 130)
print(
    "DESCRIPTIVE LINEAR TREND VS AGENT COUNT"
)
print(
    "(4 observed grid sizes only; "
    "not an asymptotic complexity claim)"
)
print("=" * 130)

print(
    f"{'Algo':<7}"
    f"{'Metric':<32}"
    f"{'Slope / agent':>18}"
    f"{'R^2':>12}"
)

print("-" * 75)


for row in trend_rows:

    print(
        f"{row['algorithm'].upper():<7}"
        f"{row['metric']:<32}"
        f"{row['slope_per_additional_agent']:>18.5f}"
        f"{row['r_squared']:>12.5f}"
    )


# ============================================================
# MODEL-SIZE CONSISTENCY
# ============================================================

print()
print("=" * 130)
print(
    "MODEL SIZE PER AGENT"
)
print("=" * 130)

for algorithm in ALGORITHMS:

    rows = sorted(
        [
            row
            for row in summary_rows
            if row["algorithm"]
            == algorithm
        ],
        key=lambda row:
            row["agents"],
    )

    values = [
        row[
            "model_size_mb_per_agent"
        ]
        for row in rows
    ]

    print(
        f"{algorithm.upper()}: "
        f"mean={mean(values):.5f} MB/agent, "
        f"range="
        f"{min(values):.5f}"
        f"–"
        f"{max(values):.5f}"
    )


print()
print("=" * 130)
print("VALIDATION")
print("=" * 130)

print(
    "Training rows:",
    len(training),
)

print(
    "Compute summary rows:",
    len(summary_rows),
)

print(
    "Trend rows:",
    len(trend_rows),
)

print()
print(
    "Summary CSV:"
)
print(summary_csv)

print()
print(
    "Trend CSV:"
)
print(trend_csv)

print()
print(
    "PASS: computational scalability "
    "analysis completed."
)
