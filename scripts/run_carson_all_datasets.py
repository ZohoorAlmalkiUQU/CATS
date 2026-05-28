from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"
    python_executable = sys.executable

    # =========================
    # CONTROL VARIABLES
    # =========================
    TARGET_EXPERIMENT = "core_experiments"
    RUN_NAME = "run_001"
    RESUME = True  # Set to True to resume from last checkpoint (if supported by config)

    # Run CARSON only on all datasets, one after another
    TARGET_DATASETS = [
        # "mnist",
        # "sst2",
        # "speech_commands",
        "cifar10",
    ]

    # Datasets to skip (for resuming after interruption).
    # Add any dataset that already completed, e.g. ["speech_commands"]
    SKIP_DATASETS: list[str] = []

    # base path:
    # configs/core_experiments/<dataset>/
    base_config_path = project_root / "configs" / TARGET_EXPERIMENT

    # =========================
    # CONFIG PATHS
    # =========================
    config_paths = [
        base_config_path
        / dataset_name
        / f"train_carson_{dataset_name}_{RUN_NAME}.yaml"
        for dataset_name in TARGET_DATASETS
    ]

    # =========================
    # VALIDATION
    # =========================
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    missing_configs = [path for path in config_paths if not path.exists()]
    if missing_configs:
        print("=" * 100)
        print("Missing config files:")
        for path in missing_configs:
            print(f"  - {path}")
        print("=" * 100)
        raise FileNotFoundError(
            "One or more CARSON run_002 config files are missing. "
            "Create them first or update TARGET_DATASETS/RUN_NAME."
        )

    # =========================
    # SUMMARY
    # =========================
    print("=" * 100)
    print("CARSON-only sequential runner")
    print("=" * 100)
    print(f"Experiment type   : {TARGET_EXPERIMENT}")
    print(f"Run name          : {RUN_NAME}")
    print(f"Datasets          : {', '.join(TARGET_DATASETS)}")
    print(f"Project root      : {project_root}")
    print("=" * 100)

    # =========================
    # RUN EXPERIMENTS
    # =========================
    for idx, config_path in enumerate(config_paths, start=1):
        dataset_name = config_path.parent.name

        if dataset_name in SKIP_DATASETS:
            print("=" * 100)
            print(f"[{idx}/{len(config_paths)}] SKIPPING (in SKIP_DATASETS): {dataset_name}")
            print("=" * 100)
            continue

        print("=" * 100)
        print(f"[{idx}/{len(config_paths)}] Running CARSON on dataset: {dataset_name}")
        print(f"Config: {config_path.name}")
        print("=" * 100)

        command = [
            python_executable,
            str(train_script),
            "--config",
            str(config_path),
        ]
        if RESUME:
            command.append("--resume")

        result = subprocess.run(command, cwd=project_root)

        if result.returncode != 0:
            raise RuntimeError(
                f"CARSON experiment failed for dataset='{dataset_name}' "
                f"using config='{config_path.name}' "
                f"(exit code={result.returncode})"
            )

        print("=" * 100)
        print(f"Completed CARSON run_002 for dataset: {dataset_name}")
        print("=" * 100)

    print("=" * 100)
    print("All CARSON run_002 experiments completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()