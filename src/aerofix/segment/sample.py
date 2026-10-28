"""Draw the reports to annotate: python -m aerofix.segment.sample

400 reports from 1995-2024 (the 2025 test year is never annotated), stratified by
family x era so old terse reports and new structured ones are both represented.
Split 280 train / 60 dev / 60 test at random (seed from config). Writes
data/annotations/seg_sample.parquet; re-running gives the identical sample.
"""
from __future__ import annotations

import pandas as pd

from aerofix.config import load_config
from aerofix.segment.common import seg_tokenize

N_TOTAL, N_DEV, N_TEST = 400, 60, 60
ERAS = [(1995, 2004), (2005, 2014), (2015, 2019), (2020, 2024)]


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["annotations"] / "seg_sample.parquet"
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it on purpose to resample (annotations refer to it)")
    df = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet",
                         columns=["row_id", "file_year", "family", "text", "n_words"])
    df = df[(df.file_year <= cfg["split"]["val_year"]) & df.n_words.between(5, 120)]
    df["era"] = pd.cut(df.file_year, [e[0] - 1 for e in ERAS] + [ERAS[-1][1]], labels=[f"{a}-{b}" for a, b in ERAS])

    per_cell = N_TOTAL // (len(ERAS) * 2)  # equal share per family x era cell
    sample = df.groupby(["family", "era"], observed=True).sample(n=per_cell, random_state=cfg["seed"])
    sample = sample.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)  # shuffle order
    sample["ann_split"] = ["test"] * N_TEST + ["dev"] * N_DEV + ["train"] * (len(sample) - N_TEST - N_DEV)
    sample = sample.sample(frac=1, random_state=cfg["seed"] + 1).reset_index(drop=True)  # mix splits in queue
    sample["tokens"] = sample.text.map(seg_tokenize)
    sample[["row_id", "file_year", "family", "era", "ann_split", "text", "tokens"]].to_parquet(out, index=False)
    print(sample.groupby(["ann_split", "family"]).size().unstack(), f"\nwrote {out}")


if __name__ == "__main__":
    main()
