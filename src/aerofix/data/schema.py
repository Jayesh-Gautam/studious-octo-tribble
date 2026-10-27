"""Schema harmonisation report: which columns each year has, and how full they are.

Result (Phase 1): all 32 yearly files share one identical 76-column header, so the
column mapping is one-to-one. Fill rates still differ by year, and a column that
is technically present can be mostly placeholders.
"""
from __future__ import annotations

import pandas as pd

from aerofix.config import load_config


def header_table(years: list[int], cfg: dict | None = None) -> pd.DataFrame:
    """One row per year: number of columns, and whether the header matches the reference year."""
    cfg = cfg or load_config()
    headers = {}
    for y in years:
        with open(cfg["paths"]["raw_sdr"] / f"SDR-{y}.csv", encoding="utf-8") as f:
            headers[y] = f.readline().strip().split(",")
    ref = headers[max(years)]
    rows = []
    for y, h in headers.items():
        missing = [c for c in cfg["build"]["usecols"] if c not in h]
        rows.append({"year": y, "n_columns": len(h), "same_as_latest": h == ref,
                     "missing_used_columns": ", ".join(missing) or "-"})
    return pd.DataFrame(rows)


def fill_rates(df: pd.DataFrame, cols: list[str], placeholders: list[str]) -> pd.DataFrame:
    """Share of rows per year where each column holds a real value (not a placeholder)."""
    ph = set(placeholders)
    filled = pd.DataFrame({c: ~df[c].fillna("").str.upper().isin(ph) for c in cols})
    filled["file_year"] = df.file_year.values
    return filled.groupby("file_year").mean().round(3)
