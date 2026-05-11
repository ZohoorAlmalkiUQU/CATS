from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"
    python_executable = sys.executable

    experiment_root = project_root / "configs" / "ablation_study"

    experiments = [
        # "fixed_tau",
        # "fixed_threshold",
        # "fixed_tau_fixed_threshold",
        "no_adaptive",
        # "fixed_tau_fixed_threshold_no_adaptive",
        # "no_inhibition",
        # "no_spiking",
        # "no_positional"
    ]

    dataset_name = "cifar10"
    config_name = "train_carson_cifar10_run_002.yaml"

    for exp_name in experiments:
        config_path = experiment_root / exp_name / dataset_name / config_name

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        print("=" * 100)
        print(f"Running experiment: {exp_name}")
        print(f"Config: {config_path}")
        print("=" * 100)

        cmd = [
            python_executable,
            str(train_script),
            "--config",
            str(config_path),
        ]

        subprocess.run(cmd, cwd=project_root, check=True)

    print("=" * 100)
    print("All selected CARSON ablation experiments completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()