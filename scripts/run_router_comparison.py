from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"

    config_paths = [
        project_root / "configs" / "Core_Experiments" /"train_carson_sst2_run_001.yaml",
        project_root / "configs" / "Core_Experiments" /"train_identity_sst2_run_001.yaml",
        project_root / "configs" / "Core_Experiments" /"train_linear_sst2_run_001.yaml",
    ]

    python_executable = sys.executable

    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        print("=" * 100)
        print(f"Running router comparison experiment: {config_path.name}")
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
    print("All router comparison experiments completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()