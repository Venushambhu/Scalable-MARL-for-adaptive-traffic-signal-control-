"""
Statistical analysis for the final 2x2 demand-sensitivity experiment.

Dataset:
    100 evaluation runs
    4 scenarios
    5 controllers
    5 paired evaluation seeds (11-15)

Primary inferential procedure:
    - paired two-sided t-test
    - paired observations by identical SUMO seed
    - Holm-Bonferroni correction
    - correction family = 7 predefined controller comparisons
      within each scenario x metric

Small-sample sensitivity analysis:
    - exact two-sided paired sign-flip permutation test
    - 2^5 = 32 possible sign assignments

Effect-size / magnitude reporting:
    - mean paired difference
    - 95% CI for paired mean difference
    - Cohen's dz
    - median paired difference

IMPORTANT:
    n = 5 paired seeds.
    Inferential results must be interpreted cautiously.
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
    / "2x2_demand"
    / "combined_2x2_100_runs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_v2"
    / "analysis"
    / "2x2_demand"
    / "statistics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SCENARIOS = [
    "low",
    "medium",
    "high",
    "dynamic",
]


METRICS = [
    "completion_rate_pct",
    "avg_travel_time",
    "avg_waiting_time",
    "avg_time_loss",
    "mean_queue_length",
]


# Seven comparisons defined BEFORE significance testing.
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

    with INPUT.open(
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    assert len(rows) == 100, (
        f"Expected 100 rows, got {len(rows)}"
    )

    return rows


def get_seed_map(
    rows,
    scenario,
    controller,
    metric,
):

    subset = [
        row
        for row in rows
        if (
            row["scenario"] == scenario
            and
            row["controller_final"]
            == controller
        )
    ]

    assert len(subset) == 5, (
        f"{scenario}/{controller}: "
        f"expected 5 rows, found {len(subset)}"
    )

    result = {}

    for row in subset:

        seed = int(
            row["seed"]
        )

        assert seed not in result, (
            f"Duplicate seed {seed}: "
            f"{scenario}/{controller}"
        )

        result[seed] = float(
            row[metric]
        )

    assert sorted(
        result.keys()
    ) == EXPECTED_SEEDS, (
        f"Unexpected seeds for "
        f"{scenario}/{controller}: "
        f"{sorted(result.keys())}"
    )

    return result


def paired_arrays(
    rows,
    scenario,
    controller_a,
    controller_b,
    metric,
):

    a_map = get_seed_map(
        rows,
        scenario,
        controller_a,
        metric,
    )

    b_map = get_seed_map(
        rows,
        scenario,
        controller_b,
        metric,
    )

    a = np.array(
        [
            a_map[seed]
            for seed in EXPECTED_SEEDS
        ],
        dtype=float,
    )

    b = np.array(
        [
            b_map[seed]
            for seed in EXPECTED_SEEDS
        ],
        dtype=float,
    )

    return a, b


def holm_adjust(p_values):

    """
    Holm-Bonferroni adjusted p-values.

    Returns adjusted values in original order.
    """

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

        multiplier = (
            m - rank
        )

        candidate = (
            multiplier
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

        adjusted[
            index
        ] = (
            adjusted_sorted[
                rank
            ]
        )

    return adjusted


def exact_sign_flip_pvalue(
    differences,
):

    """
    Exact two-sided paired sign-flip permutation test.

    Test statistic:
        absolute mean paired difference

    With n=5, evaluates all 2^5 = 32
    possible sign assignments.
    """

    d = np.asarray(
        differences,
        dtype=float,
    )

    observed = abs(
        np.mean(d)
    )

    permuted = []

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

        permuted.append(
            abs(
                np.mean(
                    signed
                )
            )
        )

    permuted = np.asarray(
        permuted
    )

    # Exact randomization probability.
    p_value = np.mean(
        permuted
        >= (
            observed
            - 1e-12
        )
    )

    return float(
        p_value
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
            cohens_dz = 0.0
        else:
            cohens_dz = (
                math.inf
                if mean_diff > 0
                else -math.inf
            )

    else:

        cohens_dz = (
            mean_diff
            / sd_diff
        )

    t_critical = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    ci_low = (
        mean_diff
        - t_critical * se
    )

    ci_high = (
        mean_diff
        + t_critical * se
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
            float(cohens_dz),
    }


def metric_direction(metric):

    if metric == "completion_rate_pct":
        return "higher_better"

    return "lower_better"


def performance_interpretation(
    metric,
    mean_difference,
):

    """
    Difference is always:
        controller_a - controller_b
    """

    direction = metric_direction(
        metric
    )

    if abs(mean_difference) < 1e-12:
        return "equal_mean"

    if direction == "higher_better":

        if mean_difference > 0:
            return "controller_a_better"

        return "controller_b_better"

    if mean_difference < 0:
        return "controller_a_better"

    return "controller_b_better"


rows = read_rows()

results = []


# ============================================================
# CALCULATE RAW TESTS
# ============================================================

for scenario in SCENARIOS:

    for metric in METRICS:

        family_rows = []

        for (
            controller_a,
            controller_b,
        ) in COMPARISONS:

            a, b = paired_arrays(
                rows,
                scenario,
                controller_a,
                controller_b,
                metric,
            )

            differences = (
                a - b
            )

            t_result = (
                stats.ttest_rel(
                    a,
                    b,
                    alternative="two-sided",
                )
            )

            effect = effect_statistics(
                differences
            )

            sign_flip_p = (
                exact_sign_flip_pvalue(
                    differences
                )
            )

            row = {
                "scenario":
                    scenario,

                "metric":
                    metric,

                "controller_a":
                    controller_a,

                "controller_b":
                    controller_b,

                "n_pairs":
                    len(a),

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
                    sign_flip_p,

                "descriptive_direction":
                    performance_interpretation(
                        metric,
                        effect[
                            "mean_difference"
                        ],
                    ),
            }

            family_rows.append(
                row
            )

        # ----------------------------------------------------
        # HOLM CORRECTION
        # One family = 7 controller comparisons
        # for ONE scenario and ONE metric.
        # ----------------------------------------------------

        raw_p = [
            row[
                "paired_t_p_raw"
            ]
            for row in family_rows
        ]

        adjusted = holm_adjust(
            raw_p
        )

        for row, p_adj in zip(
            family_rows,
            adjusted,
        ):

            row[
                "paired_t_p_holm"
            ] = float(
                p_adj
            )

            row[
                "significant_holm_0_05"
            ] = (
                p_adj < ALPHA
            )

            results.append(
                row
            )


assert len(results) == (
    len(SCENARIOS)
    * len(METRICS)
    * len(COMPARISONS)
)

assert len(results) == 140


# ============================================================
# SAVE FULL RESULTS
# ============================================================

output_csv = (
    OUTPUT_DIR
    / "paired_tests_2x2_all.csv"
)

with output_csv.open(
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


# ============================================================
# SAVE SIGNIFICANT RESULTS
# ============================================================

significant_rows = [
    row
    for row in results
    if row[
        "significant_holm_0_05"
    ]
]

significant_csv = (
    OUTPUT_DIR
    / "paired_tests_2x2_holm_significant.csv"
)

with significant_csv.open(
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
        significant_rows
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 120)
print(
    "2x2 PAIRED STATISTICAL ANALYSIS"
)
print("=" * 120)

print(
    "Pairs: seeds 11, 12, 13, 14, 15"
)

print(
    "Primary test: two-sided paired t-test"
)

print(
    "Multiplicity: Holm correction across "
    "7 comparisons within each "
    "scenario x metric family"
)

print(
    "Sensitivity: exact two-sided "
    "paired sign-flip permutation test"
)

print(
    "Alpha: 0.05"
)

print()
print(
    "IMPORTANT: n=5 pairs; "
    "interpret inferential results cautiously."
)


# ============================================================
# HEADLINE METRICS
# ============================================================

headline_metrics = [
    "completion_rate_pct",
    "avg_travel_time",
    "avg_waiting_time",
    "mean_queue_length",
]


for scenario in SCENARIOS:

    print()
    print()
    print("#" * 120)
    print(
        f"{scenario.upper()}"
    )
    print("#" * 120)

    for metric in headline_metrics:

        print()
        print(
            f"METRIC: {metric}"
        )

        print("-" * 120)

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
                row["scenario"]
                == scenario
                and
                row["metric"]
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
                f"["
                f"{row['ci95_difference_low']:.3f}, "
                f"{row['ci95_difference_high']:.3f}"
                f"]"
            )

            dz = row[
                "cohens_dz"
            ]

            if math.isinf(dz):
                dz_string = (
                    "+inf"
                    if dz > 0
                    else "-inf"
                )
            else:
                dz_string = (
                    f"{dz:.3f}"
                )

            print(
                f"{comparison:<34}"
                f"{row['mean_difference_a_minus_b']:>12.3f}"
                f"{ci:>27}"
                f"{dz_string:>10}"
                f"{row['paired_t_p_raw']:>11.4f}"
                f"{row['paired_t_p_holm']:>11.4f}"
                f"{row['sign_flip_exact_p']:>11.4f}"
                f"{str(row['significant_holm_0_05']):>11}"
            )


print()
print("=" * 120)
print(
    "HOLM-SIGNIFICANT RESULT COUNT"
)
print("=" * 120)

print(
    f"{len(significant_rows)} / "
    f"{len(results)} comparisons"
)

print()
print(
    "Full results:"
)
print(output_csv)

print()
print(
    "Holm-significant subset:"
)
print(significant_csv)

print()
print(
    "PASS: paired statistical analysis completed."
)
