from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"
    python_executable = sys.executable

    # =========================
    # 🔧 CONTROL VARIABLES
    # =========================
    TARGET_EXPERIMENT = "core_experiments"  # future: "ablation", "scaling", ...
    TARGET_DATASET = "cifar10"  # options: "sst2", "speech_commands", "cifar10"

    # base path
    base_config_path = project_root / "configs" / TARGET_EXPERIMENT

    # =========================
    # CONFIG GROUPS
    # =========================
    config_groups = {
        "sst2": [
            base_config_path / "sst2" / "train_carson_sst2_run_001.yaml",
            base_config_path / "sst2" / "train_identity_sst2_run_001.yaml",
            base_config_path / "sst2" / "train_linear_sst2_run_001.yaml",
        ],
        "speech_commands": [
            base_config_path / "speech_commands" / "train_carson_speech_commands_run_001.yaml",
            base_config_path / "speech_commands" / "train_identity_speech_commands_run_001.yaml",
            base_config_path / "speech_commands" / "train_linear_speech_commands_run_001.yaml",
        ],
        "cifar10": [
            base_config_path / "cifar10" / "train_carson_cifar10_run_001.yaml",
            base_config_path / "cifar10" / "train_identity_cifar10_run_001.yaml",
            base_config_path / "cifar10" / "train_linear_cifar10_run_001.yaml",
        ],
    }

    # =========================
    # VALIDATION
    # =========================
    if TARGET_DATASET not in config_groups:
        raise ValueError(
            f"Invalid TARGET_DATASET: {TARGET_DATASET}. "
            f"Choose from: {list(config_groups.keys())}"
        )

    selected_config_paths = config_groups[TARGET_DATASET]

    print("=" * 100)
    print(f"Experiment type   : {TARGET_EXPERIMENT}")
    print(f"Dataset           : {TARGET_DATASET}")
    print("=" * 100)

    # =========================
    # RUN EXPERIMENTS
    # =========================
    for config_path in selected_config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        print("=" * 100)
        print(f"Running: {config_path.name}")
        print("=" * 100)

        command = [
            python_executable,
            str(train_script),
            "--config",
            str(config_path),
        ]

        result = subprocess.run(command, cwd=project_root)

        if result.returncode != 0:
            raise RuntimeError(
                f"Experiment failed for config: {config_path.name} "
                f"(exit code={result.returncode})"
            )

    print("=" * 100)
    print("All experiments completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()