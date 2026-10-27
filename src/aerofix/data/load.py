"""Read one year of SDR CSV, keep only A320-family / 737 rows, add aircraft fields.

Memory: a full year is ~60k rows x 76 columns. We read only the configured
``usecols`` as strings (no type guessing) and filter immediately, so each year
stays well under 1 GB.
"""
from __future__ import annotations

import pandas as pd

from aerofix.config import load_config
from aerofix.data.aircraft import family_of, parse_variant


def read_year(year: int, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (filtered rows, raw-file stats) for one SDR year."""
    cfg = cfg or load_config()
    path = cfg["paths"]["raw_sdr"] / f"SDR-{year}.csv"
    raw = pd.read_csv(path, usecols=cfg["build"]["usecols"], dtype=str,
                      encoding="utf-8", keep_default_na=False)
    for c in raw.columns:
        raw[c] = raw[c].str.strip()

    fam = [family_of(mk, md, cfg) for mk, md in zip(raw.AircraftMake, raw.AircraftModel)]
    raw["family"] = fam
    df = raw[raw.family.notna()].copy()

    parsed = [parse_variant(mk, md, cfg) for mk, md in zip(df.AircraftMake, df.AircraftModel)]
    df["variant"] = [p[0] for p in parsed]
    df["generation"] = [p[1] for p in parsed]
    df["file_year"] = year
    for c in ("AircraftTotalTime", "AircraftTotalCycles"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["difficulty_date"] = pd.to_datetime(df.DifficultyDate, format="%m/%d/%Y", errors="coerce")

    stats = {
        "year": year,
        "total_rows": len(raw),
        "b737_rows": int((df.family == "B737").sum()),
        "a320_rows": int((df.family == "A320").sum()),
        "ata2_chapters": int(df.JASCCode.str[:2][df.JASCCode != ""].nunique()),
        "jasc_codes": int(df.JASCCode[df.JASCCode != ""].nunique()),
    }
    return df.reset_index(drop=True), stats
