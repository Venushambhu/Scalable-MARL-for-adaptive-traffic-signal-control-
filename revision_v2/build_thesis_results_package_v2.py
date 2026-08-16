"""
Build thesis-ready tables and figures from the frozen Revision V2 results.

IMPORTANT:
- Does NOT modify experimental results.
- Verifies source CSVs against FINAL_RESULTS_MANIFEST.txt.
- Generates only derived thesis assets.

Output:
    results_v2/thesis_package/
        tables/
        figures/
        DERIVED_PACKAGE_MANIFEST.txt
"""

import csv
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results_v2"

MANIFEST = (
    RESULTS
    / "FINAL_RESULTS_MANIFEST.txt"
)

PACKAGE = (
    RESULTS
    / "thesis_package"
)

TABLE_DIR = PACKAGE / "tables"
FIGURE_DIR = PACKAGE / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# AUTHORITATIVE SOURCES
# ============================================================

DEMAND_SUMMARY = (
    RESULTS
    / "analysis"
    / "2x2_demand"
    / "summary_2x2_demand.csv"
)

SCALABILITY_SUMMARY = (
    RESULTS
    / "analysis"
    / "medium_scalability"
    / "summary_medium_scalability.csv"
)

COMPUTE_SUMMARY = (
    RESULTS
    / "analysis"
    / "computational_scalability"
    / "computational_scalability_summary.csv"
)

SCALABILITY_STATS = (
    RESULTS
    / "analysis"
    / "medium_scalability"
    / "statistics"
    / "paired_tests_medium_scalability_all.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

SCENARIOS = [
    "low",
    "medium",
    "high",
    "dynamic",
]

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

CONTROLLER_LABELS = {
    "original_fixed": "Original fixed",
    "webster_fixed": "Webster",
    "actuated": "Actuated",
    "dqn_v2": "DQN",
    "ppo_v2": "PPO",
}

ALGORITHM_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
    "dqn_v2": "DQN",
    "ppo_v2": "PPO",
}


# ============================================================
# FILE UTILITIES
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


# ============================================================
# VERIFY FROZEN SOURCES
# ============================================================

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest missing: {MANIFEST}"
    )


manifest_hashes = {}

for line in MANIFEST.read_text().splitlines():

    parts = line.split(
        "  ",
        1,
    )

    if (
        len(parts) == 2
        and len(parts[0]) == 64
    ):

        manifest_hashes[
            parts[1]
        ] = parts[0]


sources = [
    DEMAND_SUMMARY,
    SCALABILITY_SUMMARY,
    COMPUTE_SUMMARY,
    SCALABILITY_STATS,
]


print()
print("=" * 88)
print("VERIFYING FROZEN SOURCE DATA")
print("=" * 88)


for source in sources:

    relative = str(
        source.relative_to(
            PROJECT_ROOT
        )
    )

    expected = manifest_hashes.get(
        relative
    )

    if expected is None:

        raise RuntimeError(
            f"{relative} is not recorded "
            "in FINAL_RESULTS_MANIFEST.txt"
        )

    actual = sha256(
        source
    )

    if actual != expected:

        raise RuntimeError(
            f"CHECKSUM MISMATCH:\n{relative}"
        )

    print(
        f"PASS: {relative}"
    )


print(
    "\nPASS: all thesis package inputs "
    "match frozen checksums."
)


# ============================================================
# LOAD DATA
# ============================================================

demand_rows = read_csv(
    DEMAND_SUMMARY
)

scale_rows = read_csv(
    SCALABILITY_SUMMARY
)

compute_rows = read_csv(
    COMPUTE_SUMMARY
)

stats_rows = read_csv(
    SCALABILITY_STATS
)


assert len(demand_rows) == 20
assert len(scale_rows) == 20
assert len(compute_rows) == 8
assert len(stats_rows) == 140


# ============================================================
# HELPERS
# ============================================================

def demand_row(
    scenario,
    controller,
):

    return next(
        row
        for row in demand_rows
        if (
            row["scenario"] == scenario
            and
            row["controller"]
            == controller
        )
    )


def scale_row(
    grid,
    controller,
):

    return next(
        row
        for row in scale_rows
        if (
            row["grid"] == grid
            and
            row["controller"]
            == controller
        )
    )


def stat_row(
    grid,
    algorithm,
    metric,
):

    return next(
        row
        for row in stats_rows
        if (
            row["grid"] == grid
            and
            row["controller_a"]
            == algorithm
            and
            row["controller_b"]
            == "actuated"
            and
            row["metric"]
            == metric
        )
    )


def latex_escape(text):

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


def save_figure(
    fig,
    basename,
):

    pdf = (
        FIGURE_DIR
        / f"{basename}.pdf"
    )

    png = (
        FIGURE_DIR
        / f"{basename}.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# TABLE 1 — DEMAND SENSITIVITY
# ============================================================

path = (
    TABLE_DIR
    / "table_demand_sensitivity.tex"
)

with path.open("w") as f:

    f.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Traffic-control performance "
        "across demand scenarios on the 2$\\times$2 network. "
        "Values are five-seed means.}\n"
        "\\label{tab:demand_sensitivity_v2}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\hline\n"
        "Scenario & Controller & Completion (\\%) "
        "& Travel (s) & Waiting (s) & Queue \\\\\n"
        "\\hline\n"
    )

    for scenario in SCENARIOS:

        for controller in CONTROLLERS:

            row = demand_row(
                scenario,
                controller,
            )

            f.write(
                f"{scenario.capitalize()} & "
                f"{CONTROLLER_LABELS[controller]} & "
                f"{float(row['completion_rate_pct_mean']):.2f} & "
                f"{float(row['avg_travel_time_mean']):.2f} & "
                f"{float(row['avg_waiting_time_mean']):.2f} & "
                f"{float(row['mean_queue_length_mean']):.2f} "
                "\\\\\n"
            )

        f.write(
            "\\hline\n"
        )

    f.write(
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ============================================================
# TABLE 2 — MEDIUM SCALABILITY
# ============================================================

path = (
    TABLE_DIR
    / "table_medium_scalability.tex"
)

with path.open("w") as f:

    f.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Medium-demand traffic-control performance "
        "across network sizes. Values are five-seed means.}\n"
        "\\label{tab:medium_scalability_v2}\n"
        "\\small\n"
        "\\begin{tabular}{rrlrrrr}\n"
        "\\hline\n"
        "Grid & Agents & Controller & Completion (\\%) "
        "& Travel (s) & Waiting (s) & Queue \\\\\n"
        "\\hline\n"
    )

    for grid in GRIDS:

        for controller in CONTROLLERS:

            row = scale_row(
                grid,
                controller,
            )

            f.write(
                f"{grid} & "
                f"{AGENTS[grid]} & "
                f"{CONTROLLER_LABELS[controller]} & "
                f"{float(row['completion_rate_pct_mean']):.2f} & "
                f"{float(row['avg_travel_time_mean']):.2f} & "
                f"{float(row['avg_waiting_time_mean']):.2f} & "
                f"{float(row['mean_queue_length_mean']):.3f} "
                "\\\\\n"
            )

        f.write(
            "\\hline\n"
        )

    f.write(
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ============================================================
# TABLE 3 — COMPUTATIONAL SCALABILITY
# ============================================================

path = (
    TABLE_DIR
    / "table_computational_scalability.tex"
)

ordered_compute = sorted(
    compute_rows,
    key=lambda row: (
        row["algorithm"],
        int(row["agents"]),
    ),
)

with path.open("w") as f:

    f.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Computational scalability of the "
        "independent DQN and PPO controllers.}\n"
        "\\label{tab:computational_scalability_v2}\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "Algorithm & Agents & Train (s) & Peak RAM (MB) "
        "& Model (MB) & Joint latency (ms) & Budget (\\%) \\\\\n"
        "\\hline\n"
    )

    for row in ordered_compute:

        f.write(
            f"{ALGORITHM_LABELS[row['algorithm']]} & "
            f"{int(row['agents'])} & "
            f"{float(row['training_runtime_s']):.2f} & "
            f"{float(row['training_peak_ram_mb']):.2f} & "
            f"{float(row['model_size_total_mb']):.3f} & "
            f"{float(row['mean_joint_decision_ms']):.4f} & "
            f"{float(row['decision_budget_pct']):.5f} "
            "\\\\\n"
        )

    f.write(
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ============================================================
# TABLE 4 — LEARNED VS ACTUATED STATISTICS
# ============================================================

path = (
    TABLE_DIR
    / "table_learned_vs_actuated_statistics.tex"
)

with path.open("w") as f:

    f.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Paired comparison of learned controllers "
        "against actuated control under medium demand. "
        "Differences are learned minus actuated; "
        "Holm-adjusted paired $t$-test $p$-values are reported.}\n"
        "\\label{tab:learned_vs_actuated_stats_v2}\n"
        "\\scriptsize\n"
        "\\begin{tabular}{rrlrrrrrrrr}\n"
        "\\hline\n"
        "Grid & Agents & Alg. "
        "& $\\Delta$Comp & $p_H$ "
        "& $\\Delta$Travel & $p_H$ "
        "& $\\Delta$Wait & $p_H$ "
        "& $\\Delta$Queue & $p_H$ \\\\\n"
        "\\hline\n"
    )

    for grid in GRIDS:

        for algorithm in [
            "dqn_v2",
            "ppo_v2",
        ]:

            comp = stat_row(
                grid,
                algorithm,
                "completion_rate_pct",
            )

            travel = stat_row(
                grid,
                algorithm,
                "avg_travel_time",
            )

            waiting = stat_row(
                grid,
                algorithm,
                "avg_waiting_time",
            )

            queue = stat_row(
                grid,
                algorithm,
                "mean_queue_length",
            )

            f.write(
                f"{grid} & "
                f"{AGENTS[grid]} & "
                f"{ALGORITHM_LABELS[algorithm]} & "
                f"{float(comp['mean_difference_a_minus_b']):+.3f} & "
                f"{float(comp['paired_t_p_holm']):.4f} & "
                f"{float(travel['mean_difference_a_minus_b']):+.3f} & "
                f"{float(travel['paired_t_p_holm']):.4f} & "
                f"{float(waiting['mean_difference_a_minus_b']):+.3f} & "
                f"{float(waiting['paired_t_p_holm']):.4f} & "
                f"{float(queue['mean_difference_a_minus_b']):+.3f} & "
                f"{float(queue['paired_t_p_holm']):.4f} "
                "\\\\\n"
            )

        f.write(
            "\\hline\n"
        )

    f.write(
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ============================================================
# FIGURE SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


# ============================================================
# DEMAND FIGURES
# ============================================================

def demand_grouped_bar(
    metric,
    ylabel,
    basename,
):

    x = np.arange(
        len(SCENARIOS)
    )

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(7.2, 4.3)
    )

    for index, controller in enumerate(
        CONTROLLERS
    ):

        values = [
            float(
                demand_row(
                    scenario,
                    controller,
                )[metric]
            )
            for scenario in SCENARIOS
        ]

        offset = (
            index
            - (
                len(CONTROLLERS)
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=(
                CONTROLLER_LABELS[
                    controller
                ]
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Low",
            "Medium",
            "High",
            "Dynamic",
        ]
    )

    ax.set_xlabel(
        "Demand scenario"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        basename,
    )


demand_grouped_bar(
    "completion_rate_pct_mean",
    "Completion rate (%)",
    "fig_demand_completion",
)

demand_grouped_bar(
    "avg_travel_time_mean",
    "Average travel time (s)",
    "fig_demand_travel",
)

demand_grouped_bar(
    "avg_waiting_time_mean",
    "Average waiting time (s)",
    "fig_demand_waiting",
)


# ============================================================
# TRAFFIC SCALABILITY FIGURES
# ============================================================

def scalability_line(
    metric,
    ylabel,
    basename,
):

    x = [
        AGENTS[grid]
        for grid in GRIDS
    ]

    fig, ax = plt.subplots(
        figsize=(6.6, 4.2)
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
        "P",
    ]

    for controller, marker in zip(
        CONTROLLERS,
        markers,
    ):

        y = [
            float(
                scale_row(
                    grid,
                    controller,
                )[metric]
            )
            for grid in GRIDS
        ]

        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.6,
            label=(
                CONTROLLER_LABELS[
                    controller
                ]
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xlabel(
        "Number of agents"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        basename,
    )


scalability_line(
    "completion_rate_pct_mean",
    "Completion rate (%)",
    "fig_scalability_completion",
)

scalability_line(
    "avg_travel_time_mean",
    "Average travel time (s)",
    "fig_scalability_travel",
)

scalability_line(
    "avg_waiting_time_mean",
    "Average waiting time (s)",
    "fig_scalability_waiting",
)


# ============================================================
# COMPUTATIONAL SCALABILITY FIGURES
# ============================================================

def compute_line(
    field,
    ylabel,
    basename,
):

    fig, ax = plt.subplots(
        figsize=(6.4, 4.1)
    )

    for algorithm, marker in [
        ("dqn", "o"),
        ("ppo", "s"),
    ]:

        rows = sorted(
            [
                row
                for row in compute_rows
                if row["algorithm"]
                == algorithm
            ],
            key=lambda row:
                int(row["agents"]),
        )

        x = [
            int(row["agents"])
            for row in rows
        ]

        y = [
            float(
                row[field]
            )
            for row in rows
        ]

        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.8,
            label=(
                ALGORITHM_LABELS[
                    algorithm
                ]
            ),
        )

    ax.set_xticks(
        [
            4,
            9,
            16,
            25,
        ]
    )

    ax.set_xlabel(
        "Number of agents"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        basename,
    )


compute_line(
    "training_runtime_s",
    "Training runtime (s)",
    "fig_compute_training_runtime",
)

compute_line(
    "training_peak_ram_mb",
    "Peak training RAM (MB)",
    "fig_compute_training_ram",
)

compute_line(
    "model_size_total_mb",
    "Total model size (MB)",
    "fig_compute_model_size",
)

compute_line(
    "mean_joint_decision_ms",
    "Mean joint decision latency (ms)",
    "fig_compute_decision_latency",
)


# ============================================================
# PACKAGE INDEX
# ============================================================

index = (
    PACKAGE
    / "RESULTS_PACKAGE_INDEX.txt"
)

index.write_text(
    """THESIS REVISION V2 — RESULTS PACKAGE

TABLES
------
table_demand_sensitivity.tex
    2x2 low/medium/high/dynamic performance.

table_medium_scalability.tex
    2x2 through 5x5 medium-demand performance.

table_computational_scalability.tex
    Training/runtime/memory/model/inference scaling.

table_learned_vs_actuated_statistics.tex
    DQN/PPO vs actuated Holm-adjusted paired statistics.


FIGURES
-------
fig_demand_completion
fig_demand_travel
fig_demand_waiting

fig_scalability_completion
fig_scalability_travel
fig_scalability_waiting

fig_compute_training_runtime
fig_compute_training_ram
fig_compute_model_size
fig_compute_decision_latency

Each figure is provided as:
    PDF  - preferred vector format for thesis
    PNG  - 300 dpi preview/fallback

All files are derived from checksum-verified frozen results.
"""
)


# ============================================================
# DERIVED PACKAGE MANIFEST
# ============================================================

generated = sorted(
    [
        path
        for path in PACKAGE.rglob("*")
        if (
            path.is_file()
            and path.name
            != "DERIVED_PACKAGE_MANIFEST.txt"
        )
    ]
)

derived_manifest = (
    PACKAGE
    / "DERIVED_PACKAGE_MANIFEST.txt"
)

with derived_manifest.open("w") as f:

    f.write(
        "THESIS REVISION V2 — "
        "DERIVED RESULTS PACKAGE MANIFEST\n"
    )

    f.write(
        "=" * 88
        + "\n"
    )

    f.write(
        "Sources verified against "
        "FINAL_RESULTS_MANIFEST.txt\n\n"
    )

    for path in generated:

        f.write(
            f"{sha256(path)}  "
            f"{path.relative_to(PACKAGE)}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

tables = sorted(
    TABLE_DIR.glob("*.tex")
)

pdf_figures = sorted(
    FIGURE_DIR.glob("*.pdf")
)

png_figures = sorted(
    FIGURE_DIR.glob("*.png")
)


print()
print("=" * 88)
print("THESIS RESULTS PACKAGE CREATED")
print("=" * 88)

print(
    "LaTeX tables:",
    len(tables),
)

print(
    "PDF figures:",
    len(pdf_figures),
)

print(
    "PNG figures:",
    len(png_figures),
)

print()
print(
    "Package:"
)
print(PACKAGE)

print()
print(
    "Derived manifest:"
)
print(derived_manifest)

print()
print(
    "PASS: thesis-ready results package "
    "generated only from frozen evidence."
)
