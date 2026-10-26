"""Seed-setting utility. Call set_seed() at the top of every training script."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility.

    Covers Python's random, NumPy, and (when torch is available) PyTorch.
    Also sets PYTHONHASHSEED so dict/set ordering is deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
