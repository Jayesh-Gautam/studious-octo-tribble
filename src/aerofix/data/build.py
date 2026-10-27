"""Build the processed dataset: python -m aerofix.data.build [--force]

Steps
1. Per year: read, filter to A320/737, add family/variant -> data/interim/sdr_<year>.parquet
2. Check the raw 2025 counts against the project brief (exit check)
3. Keep rows with a valid 4-digit JASC code and some text; derive ata2 / ata4
4. Clean text, strip boilerplate (-> context), extract + mask manual references
5. Deduplicate (same operator + same normalised text), keeping the EARLIEST copy,
   so a later-year duplicate of a training report can never reach val/test
6. Time-based split -> data/processed/sdr_a320_b737.parquet
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from aerofix.config import load_config
from aerofix.data.load import read_year
from aerofix.data.schema import fill_rates, header_table
from aerofix.preprocess.clean import basic_clean, dedup_key, strip_boilerplate
from aerofix.preprocess.refs import extract_refs, mask_refs
from aerofix.utils import get_logger, md_table

log = get_logger(__name__)


def assign_split(year: int, s: dict) -> str:
    if year <= s["train_max_year"]:
        return "train"
    if year == s["val_year"]:
        return "val"
    if year == s["test_year"]:
        return "test"
    return "unused"


def load_all(cfg: dict, force: bool) -> tuple[pd.DataFrame, list[dict]]:
    interim = cfg["paths"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    frames, stats = [], []
    y0, y1 = cfg["build"]["years"]
    for y in range(y0, y1 + 1):
        pq, js = interim / f"sdr_{y}.parquet", interim / f"sdr_{y}_stats.json"
        if pq.exists() and js.exists() and not force:  # cached: re-runs take seconds
            df, st = pd.read_parquet(pq), json.loads(js.read_text())
        else:
            df, st = read_year(y, cfg)
            df.to_parquet(pq, index=False)
            js.write_text(json.dumps(st))
        log.info("%d: %6d raw rows -> %5d B737 + %5d A320", y, st["total_rows"], st["b737_rows"], st["a320_rows"])
        frames.append(df)
        stats.append(st)
    return pd.concat(frames, ignore_index=True), stats


def check_2025(stats: list[dict], cfg: dict) -> dict:
    """Exit check: our 2025 counts must equal the facts in the project brief."""
    got = next(s for s in stats if s["year"] == 2025)
    expected = cfg["sanity_2025"]
    result = {k: {"expected": v, "got": got[k], "ok": got[k] == v} for k, v in expected.items()}
    for k, r in result.items():
        log.info("2025 check %-14s expected %6d got %6d %s", k, r["expected"], r["got"], "OK" if r["ok"] else "MISMATCH")
    if not all(r["ok"] for r in result.values()):
        raise AssertionError(f"2025 sanity check failed: {result}")
    return result


def transform(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    counts = {"filtered_rows": len(df)}

    valid_jasc = df.JASCCode.str.fullmatch(r"\d{4}")
    counts["dropped_bad_jasc"] = int((~valid_jasc).sum())
    df = df[valid_jasc].copy()
    df["ata4"] = df.JASCCode
    df["ata2"] = df.JASCCode.str[:2]

    # Config values are passed in once; per-row config reads would re-parse the YAML 600k times.
    prefixes, token = cfg["preprocess"]["boilerplate_prefixes"], cfg["preprocess"]["mask_token"]
    df["text_clean"] = df.Discrepancy.map(basic_clean)
    stripped = df.text_clean.map(lambda t: strip_boilerplate(t, prefixes))
    df["text"] = [t for t, _ in stripped]
    df["context"] = [c for _, c in stripped]
    df["n_words"] = df.text.str.split().str.len().fillna(0).astype(int)

    too_short = df.n_words < 3  # nothing to learn from "SEE ABOVE" / blank
    counts["dropped_short_text"] = int(too_short.sum())
    df = df[~too_short].copy()

    df["refs"] = df.text.map(extract_refs)
    df["text_masked"] = df.text.map(lambda t: mask_refs(t, token))

    # Earliest first, so drop_duplicates keeps the original report.
    df = df.sort_values(["file_year", "difficulty_date", "OperatorControlNumber"], kind="stable")
    key = [dedup_key(o, t) for o, t in zip(df.OperatorDesignator, df.text)]
    df["dedup_key"] = key
    dup = df.duplicated("dedup_key", keep="first")
    first_year = df.groupby("dedup_key").file_year.transform("min")
    counts["dropped_duplicates"] = int(dup.sum())
    counts["dropped_cross_year_duplicates"] = int((dup & (df.file_year != first_year)).sum())
    df = df[~dup].copy()

    df["split"] = df.file_year.map(lambda y: assign_split(y, cfg["split"]))
    date_year = df.difficulty_date.dt.year
    counts["date_year_differs_from_file_year"] = int((date_year.notna() & (date_year != df.file_year)).sum())
    counts["final_rows"] = len(df)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # stable key to join later derived tables (e.g. sdr_norm.parquet)
    return df, counts


def write_schema_report(raw: pd.DataFrame, cfg: dict) -> None:
    y0, y1 = cfg["build"]["years"]
    headers = header_table(list(range(y0, y1 + 1)), cfg)
    cols = [c for c in cfg["build"]["usecols"] if c not in ("AircraftTotalTime", "AircraftTotalCycles")]
    fills = fill_rates(raw, cols, cfg["build"]["placeholders"])
    out = cfg["paths"]["reports"] / "schema_mapping.md"
    out.write_text(
        "# SDR schema mapping\n\n"
        f"All {len(headers)} yearly files: identical header = {bool(headers.same_as_latest.all())}. "
        "Column mapping is therefore one-to-one (no renaming needed).\n\n"
        "## Headers per year\n\n" + md_table(headers, index=False) +
        "\n\n## Fill rate per year (A320 + B737 rows; placeholders such as UNK, -, blank count as missing)\n\n"
        + md_table(fills.T.rename_axis("column")) + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-read CSVs even if interim Parquet exists")
    args = ap.parse_args()
    cfg = load_config()
    cfg["paths"]["reports"].mkdir(parents=True, exist_ok=True)

    raw, stats = load_all(cfg, args.force)
    sanity = check_2025(stats, cfg)
    write_schema_report(raw, cfg)

    df, counts = transform(raw, cfg)
    counts["rows_per_split"] = df.split.value_counts().to_dict()
    for k, v in counts.items():
        log.info("%s: %s", k, v)

    cfg["paths"]["processed"].mkdir(parents=True, exist_ok=True)
    out = cfg["paths"]["processed"] / "sdr_a320_b737.parquet"
    df.to_parquet(out, index=False)
    (cfg["paths"]["reports"] / "build_stats.json").write_text(
        json.dumps({"per_year": stats, "sanity_2025": sanity, "counts": counts}, indent=2, default=str))
    log.info("wrote %s (%d rows, %.0f MB)", out, len(df), out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
