"""
Run all SNN classifier readout variants on the full CARSON model.

Each run keeps all CARSON settings identical (router, LIF encoder, hidden dim,
optimizer, lr, batch size, epochs, seed, early stopping, regularization).
Only the SNN head readout type changes.

Usage
-----
    python scripts/run_snn_readout_study.py [--resume] [--dataset DATASET]

Options
-------
--resume    Skip completed checkpoints and continue from the last saved state.
--dataset   Dataset to run on (default: speech_commands).

Readout variants
----------------
mean_membrane          Baseline SNN head — average membrane over valid timesteps.
last_membrane          Final membrane state only.
mean_spikes            Average spike rate over valid timesteps (pure spike signal).
spike_count            Total spike count per class (strict spike-based decision).
membrane_spike_hybrid  Concatenate mean_membrane + spike_count → linear to num_classes.
temporal_pool          Soft attention over the membrane trace (learnable aggregation).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


READOUT_VARIANTS = [
    "mean_membrane",
    "last_membrane",
    "mean_spikes",
    "spike_count",
    "membrane_spike_hybrid",
    "temporal_pool",
]

CONFIG_TEMPLATE = "train_carson_{dataset}_{readout}.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SNN readout study experiments")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dataset", default="speech_commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"
    config_root = project_root / "configs" / "snn_readout_study" / args.dataset

    print("=" * 100)
    print(f"SNN Readout Study — dataset: {args.dataset}")
    print(f"Variants: {READOUT_VARIANTS}")
    print(f"Resume:   {args.resume}")
    print("=" * 100)

    for readout in READOUT_VARIANTS:
        config_name = CONFIG_TEMPLATE.format(dataset=args.dataset, readout=readout)
        config_path = config_root / config_name

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Expected configs/snn_readout_study/{args.dataset}/{config_name}"
            )

        print()
        print("=" * 100)
        print(f"Readout: {readout}")
        print(f"Config:  {config_path}")
        print("=" * 100)

        cmd = [sys.executable, str(train_script), "--config", str(config_path)]
        if args.resume:
            cmd.append("--resume")

        subprocess.run(cmd, cwd=project_root, check=True)

    print()
    print("=" * 100)
    print("All SNN readout study experiments completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
