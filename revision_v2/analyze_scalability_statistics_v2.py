"""
Paired statistical analysis for medium-demand scalability.

Dataset:
    4 grid sizes
    5 controllers
    5 paired seeds
    = 100 evaluation runs

Grid sizes:
    2x2 -> 4 agents
    3x3 -> 9 agents
    4x4 -> 16 agents
    5x5 -> 25 agents

Primary inferential procedure:
    - two-sided paired t-test
    - pairing by identical evaluation seed
    - Holm correction across seven predefined
      controller comparisons within each grid x metric

Sensitivity:
    - exact two-sided paired sign-flip test
      over 2^5 = 32 sign assignments

Effect magnitude:
    - paired mean difference
    - 95% CI
    - Cohen's dz
    - median paired difference

n = 5 pairs, therefore inferential claims
must remain cautious.
"""

import csv
import itertools
import math
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT_ROOT
    / "results_v2"
    / "analysis"
    / "medium_scalability"
    / "combined_medium_scalability_100_runs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_v2"
    / "analysis"
    / "medium_scalability"
    / "statistics"
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

METRICS = [
    "completion_rate_pct",
    "avg_travel_time",
    "avg_waiting_time",
    "avg_time_loss",
    "mean_queue_length",
]

COMPARISONS = [
    ("dqn_v2", "original_fixed"),
    ("dqn_v2", "webster_fixed"),
    ("dqn_v2", "actuated"),

    ("ppo_v2", "original_fixed"),
    ("ppo_v2", "webster_fixed"),
    ("ppo_v2", "actuated"),

    ("dqn_v2", "ppo_v2"),
]

EXPECTED_SEEDS = [
    11,
    12,
    13,
    14,
    15,
]

ALPHA = 0.05


def read_rows():

    if not INPUT.exists():

        raise FileNotFoundError(
            INPUT
        )

    with INPUT.open(
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    assert len(rows) == 100, (
        f"Expected 100 rows, "
        f"found {len(rows)}."
    )

    return rows


def seed_map(
    rows,
    grid,
    controller,
    metric,
):

    subset = [
        row
        for row in rows
        if (
            row["grid"] == grid
            and
            row["controller_final"]
            == controller
        )
    ]

    assert len(subset) == 5, (
        f"{grid}/{controller}: "
        f"expected 5 rows, "
        f"found {len(subset)}."
    )

    result = {}

    for row in subset:

        seed = int(
            row["seed"]
        )

        assert seed not in result

        result[seed] = float(
            row[metric]
        )

    assert sorted(
        result
    ) == EXPECTED_SEEDS

    return result


def paired_arrays(
    rows,
    grid,
    controller_a,
    controller_b,
    metric,
):

    a_map = seed_map(
        rows,
        grid,
        controller_a,
        metric,
    )

    b_map = seed_map(
        rows,
        grid,
        controller_b,
        metric,
    )

    a = np.array(
        [
            a_map[seed]
            for seed
            in EXPECTED_SEEDS
        ],
        dtype=float,
    )

    b = np.array(
        [
            b_map[seed]
            for seed
            in EXPECTED_SEEDS
        ],
        dtype=float,
    )

    return a, b


def holm_adjust(p_values):

    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    adjusted_sorted = np.zeros(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(
        order
    ):

        candidate = (
            (m - rank)
            * p[index]
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_sorted[
            rank
        ] = min(
            running_max,
            1.0,
        )

    adjusted = np.zeros(
        m,
        dtype=float,
    )

    for rank, index in enumerate(
        order
    ):

        adjusted[index] = (
            adjusted_sorted[
                rank
            ]
        )

    return adjusted


def exact_sign_flip(
    differences,
):

    d = np.asarray(
        differences,
        dtype=float,
    )

    observed = abs(
        np.mean(d)
    )

    statistics = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(d),
    ):

        signed = (
            d
            * np.asarray(
                signs,
                dtype=float,
            )
        )

        statistics.append(
            abs(
                np.mean(
                    signed
                )
            )
        )

    statistics = np.asarray(
        statistics
    )

    return float(
        np.mean(
            statistics
            >= (
                observed
                - 1e-12
            )
        )
    )


def effect_statistics(
    differences,
):

    d = np.asarray(
        differences,
        dtype=float,
    )

    n = len(d)

    mean_diff = float(
        np.mean(d)
    )

    median_diff = float(
        np.median(d)
    )

    sd_diff = float(
        np.std(
            d,
            ddof=1,
        )
    )

    se = (
        sd_diff
        / math.sqrt(n)
    )

    if sd_diff == 0:

        if mean_diff == 0:
            dz = 0.0
        else:
            dz = (
                math.inf
                if mean_diff > 0
                else -math.inf
            )

    else:

        dz = (
            mean_diff
            / sd_diff
        )

    critical = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    ci_low = (
        mean_diff
        - critical * se
    )

    ci_high = (
        mean_diff
        + critical * se
    )

    return {
        "mean_difference":
            mean_diff,

        "median_difference":
            median_diff,

        "sd_difference":
            sd_diff,

        "ci95_low":
            float(ci_low),

        "ci95_high":
            float(ci_high),

        "cohens_dz":
            float(dz),
    }


def direction(
    metric,
    mean_diff,
):

    # Difference:
    # controller_a - controller_b

    if abs(mean_diff) < 1e-12:
        return "equal_mean"

    if metric == (
        "completion_rate_pct"
    ):

        if mean_diff > 0:
            return "controller_a_better"

        return "controller_b_better"

    # Lower is better for all other metrics.

    if mean_diff < 0:
        return "controller_a_better"

    return "controller_b_better"


rows = read_rows()

results = []


# ============================================================
# TESTS
# ============================================================

for grid in GRIDS:

    for metric in METRICS:

        family = []

        for (
            controller_a,
            controller_b,
        ) in COMPARISONS:

            a, b = paired_arrays(
                rows,
                grid,
                controller_a,
                controller_b,
                metric,
            )

            differences = (
                a - b
            )

            t_result = stats.ttest_rel(
                a,
                b,
                alternative="two-sided",
            )

            effect = effect_statistics(
                differences
            )

            permutation_p = (
                exact_sign_flip(
                    differences
                )
            )

            family.append(
                {
                    "grid":
                        grid,

                    "agents":
                        AGENTS[grid],

                    "metric":
                        metric,

                    "controller_a":
                        controller_a,

                    "controller_b":
                        controller_b,

                    "n_pairs":
                        5,

                    "seeds":
                        "11,12,13,14,15",

                    "mean_a":
                        float(
                            np.mean(a)
                        ),

                    "mean_b":
                        float(
                            np.mean(b)
                        ),

                    "mean_difference_a_minus_b":
                        effect[
                            "mean_difference"
                        ],

                    "median_difference_a_minus_b":
                        effect[
                            "median_difference"
                        ],

                    "sd_paired_difference":
                        effect[
                            "sd_difference"
                        ],

                    "ci95_difference_low":
                        effect[
                            "ci95_low"
                        ],

                    "ci95_difference_high":
                        effect[
                            "ci95_high"
                        ],

                    "cohens_dz":
                        effect[
                            "cohens_dz"
                        ],

                    "paired_t_statistic":
                        float(
                            t_result.statistic
                        ),

                    "paired_t_p_raw":
                        float(
                            t_result.pvalue
                        ),

                    "sign_flip_exact_p":
                        permutation_p,

                    "descriptive_direction":
                        direction(
                            metric,
                            effect[
                                "mean_difference"
                            ],
                        ),
                }
            )

        raw_p = [
            row[
                "paired_t_p_raw"
            ]
            for row in family
        ]

        adjusted = holm_adjust(
            raw_p
        )

        for row, p_holm in zip(
            family,
            adjusted,
        ):

            row[
                "paired_t_p_holm"
            ] = float(
                p_holm
            )

            row[
                "significant_holm_0_05"
            ] = (
                p_holm < ALPHA
            )

            results.append(
                row
            )


expected_count = (
    len(GRIDS)
    * len(METRICS)
    * len(COMPARISONS)
)

assert expected_count == 140
assert len(results) == 140


# ============================================================
# SAVE
# ============================================================

full_output = (
    OUTPUT_DIR
    / "paired_tests_medium_scalability_all.csv"
)

with full_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        results
    )


significant = [
    row
    for row in results
    if row[
        "significant_holm_0_05"
    ]
]


sig_output = (
    OUTPUT_DIR
    / "paired_tests_medium_scalability_holm_significant.csv"
)

with sig_output.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        significant
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 122)
print(
    "MEDIUM-DEMAND SCALABILITY "
    "PAIRED STATISTICAL ANALYSIS"
)
print("=" * 122)

print(
    "Seeds: 11, 12, 13, 14, 15"
)

print(
    "Primary test: "
    "two-sided paired t-test"
)

print(
    "Multiplicity: Holm correction "
    "across seven controller comparisons "
    "within each grid x metric"
)

print(
    "Sensitivity: exact two-sided "
    "paired sign-flip test"
)

print(
    "Alpha: 0.05"
)

print(
    "IMPORTANT: only five paired seeds; "
    "interpret inferential results cautiously."
)


HEADLINE = [
    "completion_rate_pct",
    "avg_travel_time",
    "avg_waiting_time",
    "mean_queue_length",
]


for grid in GRIDS:

    print()
    print()
    print("#" * 122)

    print(
        f"{grid} | "
        f"{AGENTS[grid]} AGENTS"
    )

    print("#" * 122)

    for metric in HEADLINE:

        print()
        print(
            f"METRIC: {metric}"
        )

        print("-" * 122)

        print(
            f"{'Comparison':<34}"
            f"{'Mean diff':>12}"
            f"{'95% CI':>27}"
            f"{'dz':>10}"
            f"{'p raw':>11}"
            f"{'p Holm':>11}"
            f"{'SignFlip':>11}"
            f"{'Holm sig':>11}"
        )

        subset = [
            row
            for row in results
            if (
                row["grid"] == grid
                and row["metric"]
                == metric
            )
        ]

        for row in subset:

            comparison = (
                f"{row['controller_a']} "
                f"vs "
                f"{row['controller_b']}"
            )

            ci = (
                "["
                f"{row['ci95_difference_low']:.3f}, "
                f"{row['ci95_difference_high']:.3f}"
                "]"
            )

            dz = row[
                "cohens_dz"
            ]

            if math.isinf(dz):

                dz_text = (
                    "+inf"
                    if dz > 0
                    else "-inf"
                )

            else:

                dz_text = (
                    f"{dz:.3f}"
                )

            print(
                f"{comparison:<34}"
                f"{row['mean_difference_a_minus_b']:>12.3f}"
                f"{ci:>27}"
                f"{dz_text:>10}"
                f"{row['paired_t_p_raw']:>11.4f}"
                f"{row['paired_t_p_holm']:>11.4f}"
                f"{row['sign_flip_exact_p']:>11.4f}"
                f"{str(row['significant_holm_0_05']):>11}"
            )


# ============================================================
# LEARNED VS ACTUATED SUMMARY
# ============================================================

print()
print("=" * 122)
print(
    "LEARNED CONTROLLERS VS ACTUATED "
    "— HEADLINE SUMMARY"
)
print("=" * 122)


for grid in GRIDS:

    print()
    print(
        f"{grid} | {AGENTS[grid]} agents"
    )

    for algorithm in [
        "dqn_v2",
        "ppo_v2",
    ]:

        print(
            f"  {algorithm}:"
        )

        for metric in HEADLINE:

            row = next(
                r
                for r in results
                if (
                    r["grid"] == grid
                    and
                    r["metric"] == metric
                    and
                    r["controller_a"]
                    == algorithm
                    and
                    r["controller_b"]
                    == "actuated"
                )
            )

            print(
                f"    {metric}: "
                f"diff="
                f"{row['mean_difference_a_minus_b']:+.3f}, "
                f"Holm p="
                f"{row['paired_t_p_holm']:.4f}, "
                f"sig="
                f"{row['significant_holm_0_05']}"
            )


print()
print("=" * 122)
print(
    "HOLM-SIGNIFICANT RESULT COUNT"
)
print("=" * 122)

print(
    f"{len(significant)} / "
    f"{len(results)} comparisons"
)

print()
print(
    "Full results:"
)
print(full_output)

print()
print(
    "Holm-significant subset:"
)
print(sig_output)

print()
print(
    "PASS: medium scalability "
    "paired statistical analysis completed."
)
