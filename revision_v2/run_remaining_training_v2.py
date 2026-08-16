"""
Resume-safe Thesis Revision V2 training runner.

Runs the remaining locked 50,000-step DQN/PPO experiments.

Demand phase:
    2x2 low
    2x2 high
    2x2 dynamic

Scalability phase:
    3x3 medium
    4x4 medium
    5x5 medium

Each configuration:
    DQN + PPO
    50,000 training steps
    seed = 1

Already-completed valid experiments are skipped automatically.

The runner:
- executes each training in a fresh Python process
- preserves stdout/stderr in individual log files
- verifies training_summary.csv
- verifies expected model counts
- can safely be restarted
- never deletes previous successful work
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

RESULTS_V2 = PROJECT_ROOT / "results_v2"
MODELS_V2 = PROJECT_ROOT / "models_v2"

RUNNER_LOG_DIR = (
    RESULTS_V2
    / "runner_logs"
)

RUNNER_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_CSV = (
    RESULTS_V2
    / "training_compute_master.csv"
)

TOTAL_STEPS = 50_000
TRAINING_SEED = 1


AGENT_COUNTS = {
    "2x2": 4,
    "3x3": 9,
    "4x4": 16,
    "5x5": 25,
}


TRAINERS = {
    "dqn": HERE / "train_dqn_marl_v2.py",
    "ppo": HERE / "train_ppo_marl_v2.py",
}


DEMAND_CONFIGS = [
    ("2x2", "low"),
    ("2x2", "high"),
    ("2x2", "dynamic"),
]


SCALABILITY_CONFIGS = [
    ("3x3", "medium"),
    ("4x4", "medium"),
    ("5x5", "medium"),
]


ALL_EXPERIMENTS = [
    ("dqn", "2x2", "low"),
    ("ppo", "2x2", "low"),

    ("dqn", "2x2", "medium"),
    ("ppo", "2x2", "medium"),

    ("dqn", "2x2", "high"),
    ("ppo", "2x2", "high"),

    ("dqn", "2x2", "dynamic"),
    ("ppo", "2x2", "dynamic"),

    ("dqn", "3x3", "medium"),
    ("ppo", "3x3", "medium"),

    ("dqn", "4x4", "medium"),
    ("ppo", "4x4", "medium"),

    ("dqn", "5x5", "medium"),
    ("ppo", "5x5", "medium"),
]


def summary_path(
    algorithm,
    grid,
    scenario,
):

    return (
        RESULTS_V2
        / "training"
        / f"grid{grid}"
        / algorithm
        / scenario
        / "training_summary.csv"
    )


def model_dir(
    algorithm,
    grid,
    scenario,
):

    return (
        MODELS_V2
        / f"grid{grid}"
        / algorithm
        / scenario
    )


def read_summary(path):

    if not path.exists():
        return None

    try:

        with path.open(
            newline="",
        ) as handle:

            rows = list(
                csv.DictReader(handle)
            )

        if len(rows) != 1:
            return None

        return rows[0]

    except Exception:
        return None


def expected_models_exist(
    algorithm,
    grid,
    scenario,
):

    directory = model_dir(
        algorithm,
        grid,
        scenario,
    )

    if not directory.exists():
        return False

    files = list(
        directory.glob(
            f"{algorithm}_*_{scenario}.zip"
        )
    )

    return (
        len(files)
        == AGENT_COUNTS[grid]
    )


def experiment_complete(
    algorithm,
    grid,
    scenario,
):

    path = summary_path(
        algorithm,
        grid,
        scenario,
    )

    row = read_summary(path)

    if row is None:
        return False

    try:

        correct = (
            row.get("algorithm")
            == algorithm
            and row.get("grid")
            == grid
            and row.get("scenario")
            == scenario
            and int(row.get("total_steps", 0))
            == TOTAL_STEPS
            and int(row.get("training_seed", -1))
            == TRAINING_SEED
            and row.get("status")
            == "PASS"
        )

    except Exception:
        return False

    if not correct:
        return False

    return expected_models_exist(
        algorithm,
        grid,
        scenario,
    )


def normalized_summary(
    algorithm,
    grid,
    scenario,
):

    path = summary_path(
        algorithm,
        grid,
        scenario,
    )

    row = read_summary(path)

    if row is None:
        return None

    if algorithm == "dqn":

        buffer_type = (
            "replay_buffer"
        )

        buffer_total = (
            row.get(
                "replay_buffer_total_mb",
                "",
            )
        )

        device = (
            row.get(
                "training_device",
                "",
            )
        )

    else:

        buffer_type = (
            "rollout_buffer"
        )

        buffer_total = (
            row.get(
                "rollout_buffer_total_mb",
                "",
            )
        )

        device = (
            row.get(
                "training_device",
                "",
            )
        )

    return {
        "algorithm":
            algorithm,

        "grid":
            grid,

        "scenario":
            scenario,

        "training_seed":
            row.get(
                "training_seed",
                "",
            ),

        "agents":
            row.get(
                "agents",
                "",
            ),

        "total_steps":
            row.get(
                "total_steps",
                "",
            ),

        "episodes_completed":
            row.get(
                "episodes_completed",
                "",
            ),

        "training_runtime_s":
            row.get(
                "training_runtime_s",
                "",
            ),

        "peak_process_tree_ram_mb":
            row.get(
                "peak_process_tree_ram_mb",
                "",
            ),

        "model_size_total_mb":
            row.get(
                "model_size_total_mb",
                "",
            ),

        "buffer_type":
            buffer_type,

        "buffer_total_mb":
            buffer_total,

        "training_device":
            device,

        "status":
            row.get(
                "status",
                "",
            ),

        "summary_file":
            str(path),
    }


def write_master_csv():

    rows = []

    for (
        algorithm,
        grid,
        scenario,
    ) in ALL_EXPERIMENTS:

        row = normalized_summary(
            algorithm,
            grid,
            scenario,
        )

        if row is not None:
            rows.append(row)

    fieldnames = [
        "algorithm",
        "grid",
        "scenario",
        "training_seed",
        "agents",
        "total_steps",
        "episodes_completed",
        "training_runtime_s",
        "peak_process_tree_ram_mb",
        "model_size_total_mb",
        "buffer_type",
        "buffer_total_mb",
        "training_device",
        "status",
        "summary_file",
    ]

    with MASTER_CSV.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def jobs_for_phase(phase):

    if phase == "demand":

        configs = DEMAND_CONFIGS

    elif phase == "scalability":

        configs = SCALABILITY_CONFIGS

    elif phase == "all":

        configs = (
            DEMAND_CONFIGS
            + SCALABILITY_CONFIGS
        )

    else:
        raise ValueError(phase)

    jobs = []

    for grid, scenario in configs:

        for algorithm in [
            "dqn",
            "ppo",
        ]:

            jobs.append(
                (
                    algorithm,
                    grid,
                    scenario,
                )
            )

    return jobs


def run_job(
    algorithm,
    grid,
    scenario,
):

    trainer = TRAINERS[
        algorithm
    ]

    if not trainer.exists():

        raise FileNotFoundError(
            f"Trainer missing: "
            f"{trainer}"
        )

    log_file = (
        RUNNER_LOG_DIR
        / (
            f"{algorithm}_"
            f"grid{grid}_"
            f"{scenario}_"
            f"50k.log"
        )
    )

    command = [
        sys.executable,
        str(trainer),

        "--grid",
        grid,

        "--scenario",
        scenario,

        "--total_steps",
        str(TOTAL_STEPS),

        "--seed",
        str(TRAINING_SEED),
    ]

    print()
    print("=" * 90)

    print(
        f"STARTING: "
        f"{algorithm.upper()} | "
        f"{grid} | "
        f"{scenario}"
    )

    print("=" * 90)

    print(
        "Command:"
    )

    print(
        " ".join(command)
    )

    print(
        f"\nLog:\n{log_file}"
    )

    with log_file.open(
        "w",
    ) as log_handle:

        process = subprocess.Popen(
            command,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:

            print(
                line,
                end="",
            )

            log_handle.write(
                line
            )

            log_handle.flush()

        return_code = (
            process.wait()
        )

    if return_code != 0:

        raise RuntimeError(
            f"{algorithm} "
            f"{grid}/{scenario} "
            f"failed with exit code "
            f"{return_code}."
        )

    if not experiment_complete(
        algorithm,
        grid,
        scenario,
    ):

        raise RuntimeError(
            f"{algorithm} "
            f"{grid}/{scenario} "
            "finished but validation "
            "of its summary/models failed."
        )

    print()
    print(
        "PASS: training experiment "
        "validated."
    )


def print_plan(jobs):

    print()
    print("=" * 90)
    print(
        "THESIS REVISION V2 — "
        "TRAINING PLAN"
    )
    print("=" * 90)

    for index, (
        algorithm,
        grid,
        scenario,
    ) in enumerate(
        jobs,
        start=1,
    ):

        status = (
            "SKIP - already complete"
            if experiment_complete(
                algorithm,
                grid,
                scenario,
            )
            else
            "RUN"
        )

        print(
            f"{index:>2}. "
            f"{algorithm.upper():<4} "
            f"{grid:<4} "
            f"{scenario:<8} "
            f"{status}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=[
            "demand",
            "scalability",
            "all",
        ],
        default="demand",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    jobs = jobs_for_phase(
        args.phase
    )

    print_plan(jobs)

    write_master_csv()

    if args.dry_run:

        print()
        print(
            "DRY RUN COMPLETE. "
            "No training started."
        )

        return

    completed_this_run = 0
    skipped = 0

    try:

        for (
            algorithm,
            grid,
            scenario,
        ) in jobs:

            if experiment_complete(
                algorithm,
                grid,
                scenario,
            ):

                print()
                print(
                    f"SKIP: "
                    f"{algorithm.upper()} "
                    f"{grid}/{scenario} "
                    "already has a valid "
                    "50k PASS result."
                )

                skipped += 1

                continue

            run_job(
                algorithm,
                grid,
                scenario,
            )

            completed_this_run += 1

            write_master_csv()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "INTERRUPTED BY USER."
        )

        print(
            "Completed experiments remain "
            "saved and will be skipped "
            "when the runner is restarted."
        )

        write_master_csv()

        raise

    except Exception as exc:

        print()
        print()
        print("=" * 90)
        print("TRAINING RUNNER STOPPED")
        print("=" * 90)

        print(
            f"\nReason:\n{exc}"
        )

        print(
            "\nPreviously completed runs "
            "remain intact."
        )

        write_master_csv()

        raise

    write_master_csv()

    print()
    print("=" * 90)
    print("TRAINING PHASE COMPLETE")
    print("=" * 90)

    print(
        f"New experiments completed: "
        f"{completed_this_run}"
    )

    print(
        f"Experiments skipped:       "
        f"{skipped}"
    )

    print(
        f"\nMaster compute table:\n"
        f"{MASTER_CSV}"
    )


if __name__ == "__main__":
    main()
