"""
Seed Utility

Ensures reproducibility across experiments by fixing random seeds.

Responsibilities:

* Set seed for:

  * Python random
  * NumPy
  * PyTorch (CPU and CUDA)
* Configure deterministic behavior when needed

Why this matters:

* Ensures consistent results across runs
* Critical for fair comparison between routing methods
* Required for research reproducibility

Typical usage:

* Called once at the beginning of training scripts

Important:

* Some GPU operations may still be nondeterministic
* Deterministic mode may reduce performance

Design Philosophy:

* Keep this module simple and centralized
* Avoid scattering seed logic across the codebase
  """
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False