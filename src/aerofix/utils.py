"""Shared helpers: seeding and logging."""
from __future__ import annotations

import logging
import os
import random

import numpy as np

from aerofix.config import load_config


def set_seed(seed: int) -> None:
    """Seed every random number generator we use so runs are repeatable.

    torch is imported lazily so data-only scripts don't pay its import cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console and to logs/aerofix.log, one shared format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (module imported twice)
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    log_dir = load_config()["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "aerofix.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
