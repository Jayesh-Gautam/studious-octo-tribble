"""Load configs/config.yaml and resolve paths against the project root.

Why: every script reads settings from one file, so results are reproducible and
no absolute path (e.g. D:\\...) is ever hard-coded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/aerofix/config.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config dict, with every entry in ``paths`` turned into an
    absolute ``Path`` under the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    cfg["project_root"] = PROJECT_ROOT
    return cfg
