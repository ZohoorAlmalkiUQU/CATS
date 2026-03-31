"""
Experiment Logger

Handles logging of training progress, metrics, and diagnostics.

Responsibilities:

* Log:

  * training/validation loss
  * accuracy / F1
  * spiking metrics (firing rate, spikes per sample)
  * optional routing statistics
* Support:

  * console logging
  * file logging
  * (optional) future integration with tools like TensorBoard / WandB

Why this matters:

* Tracks model behavior over time
* Helps debug training instability
* Enables analysis of spiking activity
* Required for reporting results in the paper

Design Philosophy:

* Keep logging separate from training logic
* Provide a clean interface (e.g., log(metrics_dict))
* Allow easy extension without modifying training loop

Important:

* Logging format should be consistent across all experiments
* Do not hardcode metric names inside training scripts

Optional Extensions:

* Save logs as JSON/CSV
* Track best model checkpoints
* Add experiment identifiers (run_name, config)
  """

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    def __init__(self, save_dir: Optional[str] = None, run_name: str = "run") -> None:
        self.history = []
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.run_name = run_name

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.save_dir / f"{run_name}_metrics.jsonl"
        else:
            self.log_path = None

    def log(self, metrics: Dict[str, Any]) -> None:
        self.history.append(metrics)

        line = " | ".join(f"{k}={v}" for k, v in metrics.items())
        print(line)

        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")