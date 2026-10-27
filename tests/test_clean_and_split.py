import pandas as pd

from aerofix.config import load_config
from aerofix.data.build import assign_split, transform
from aerofix.preprocess.clean import basic_clean, dedup_key, strip_boilerplate


def test_basic_clean():
    assert basic_clean("  l1 door\tdamper  leaking ") == "L1 DOOR DAMPER LEAKING"
    assert basic_clean(None) == ""


def test_strip_boilerplate():
    assert strip_boilerplate("AIRCRAFT WAS NOT GROUNDED: L1 DOOR LEAK") == ("L1 DOOR LEAK", "NOT_GROUNDED")
    assert strip_boilerplate("L1 DOOR LEAK") == ("L1 DOOR LEAK", "NONE")


def test_dedup_key_ignores_punctuation():
    assert dedup_key("AALA", "R/H  DOOR, LEAKING") == dedup_key("aala", "R H DOOR LEAKING")


def test_assign_split():
    s = load_config()["split"]
    assert [assign_split(y, s) for y in (1995, 2023, 2024, 2025, 2026)] == ["train", "train", "val", "test", "unused"]


def _row(year, op, text, n):
    return {"file_year": year, "OperatorDesignator": op, "Discrepancy": text, "JASCCode": "5210",
            "difficulty_date": pd.Timestamp(f"{year}-01-0{n}"), "OperatorControlNumber": str(n)}


def test_duplicates_never_cross_splits():
    # Same operator writes the same text in 2023 (train) and 2025 (test).
    df = pd.DataFrame([
        _row(2023, "AALA", "L1 DOOR DAMPER LEAKING REPLACED DAMPER", 1),
        _row(2025, "AALA", "L1 DOOR DAMPER LEAKING, REPLACED DAMPER", 2),
        _row(2025, "SWAA", "L1 DOOR DAMPER LEAKING REPLACED DAMPER", 3),  # other operator: kept
    ])
    out, counts = transform(df, load_config())
    assert counts["dropped_cross_year_duplicates"] == 1
    assert sorted(zip(out.file_year, out.OperatorDesignator)) == [(2023, "AALA"), (2025, "SWAA")]
