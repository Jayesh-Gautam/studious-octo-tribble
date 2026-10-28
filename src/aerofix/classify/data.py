"""Load classification data and build label spaces.

Label spaces are fitted on TRAIN only (choosing classes by looking at val/test
would leak). Anything outside a label space becomes OTHER:
    ata2       top-K chapters                   (16 classes with OTHER)
    ata4       JASC codes with >= N train rows  (226 classes with OTHER)
    condition  top-N PartCondition codes, placeholders (UNKNOWN, NONE) excluded
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aerofix.config import load_config

TARGETS = ("ata2", "ata4", "condition")
OTHER = "OTHER"


def load_frame(cfg: dict | None = None) -> pd.DataFrame:
    """Main table + Phase 3 segments (+ Phase 4 normalised problem text if built)."""
    cfg = cfg or load_config()
    proc = cfg["paths"]["processed"]
    df = pd.read_parquet(proc / "sdr_a320_b737.parquet",
                         columns=["row_id", "split", "family", "variant", "file_year", "ata2", "ata4",
                                  "PartCondition", "text", "text_masked"])
    seg = pd.read_parquet(proc / "sdr_segments.parquet", columns=["row_id", "problem_text", "problem_masked"])
    df = df.merge(seg, on="row_id", how="left")
    norm = proc / "sdr_problem_norm.parquet"
    if norm.exists():
        df = df.merge(pd.read_parquet(norm), on="row_id", how="left")
    df["condition"] = df.PartCondition.fillna("").str.strip().str.upper()
    return df


@dataclass
class LabelSpace:
    classes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, cfg: dict) -> "LabelSpace":
        c = cfg["classify"]
        ata2 = train.ata2.value_counts().index[: c["top_k_ata2"]].tolist()
        a4 = train.ata4.value_counts()
        ata4 = a4[a4 >= c["ata4_min_count"]].index.tolist()
        cond = train.condition[~train.condition.isin(c["condition_exclude"])].value_counts()
        condition = cond.index[: c["top_n_condition"]].tolist()
        return cls({"ata2": sorted(ata2) + [OTHER], "ata4": sorted(ata4) + [OTHER],
                    "condition": sorted(condition) + [OTHER]})

    def encode(self, target: str, values: pd.Series) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes[target])}
        other = idx[OTHER]
        return np.array([idx.get(v, other) for v in values], dtype=np.int64)

    def names(self, target: str) -> list[str]:
        return self.classes[target]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.classes, indent=1))

    @classmethod
    def load(cls, path: Path) -> "LabelSpace":
        return cls(json.loads(path.read_text()))


def class_weights(y: np.ndarray, n_classes: int, power: float) -> np.ndarray:
    """(1/frequency)^power, normalised to mean 1. power=1 is fully 'balanced'; 0.5 softens it so
    very rare classes don't dominate the loss of a neural net (which made training unstable)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** power, 0.0)
    return w / w[counts > 0].mean()


def save_preds(out_dir: Path, split: str, row_ids: np.ndarray, probs: dict[str, np.ndarray]) -> None:
    """Per-split class probabilities for every target -> preds_<split>.npz (report.py reads these)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"preds_{split}.npz", row_id=row_ids,
                        **{t: p.astype(np.float16) for t, p in probs.items()})
