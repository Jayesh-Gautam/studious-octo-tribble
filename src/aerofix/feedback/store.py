"""Technician feedback store (SQLite, one file: data/feedback/feedback.sqlite).

Each row records what AeroFix predicted and what the technician said:
was the predicted ATA chapter correct (and if not, the right one), which
recommended action actually fixed the fault, and an optional comment.
Phase 9 turns these rows into extra training examples.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from aerofix.config import load_config

SCHEMA = """CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'app',      -- 'app' or 'simulated' (Phase 9)
    family TEXT, variant TEXT, stage TEXT,
    text TEXT NOT NULL,
    predicted_ata2 TEXT, predicted_condition TEXT,
    system_correct INTEGER,                  -- 1 yes, 0 no, NULL not answered
    correct_ata2 TEXT,                       -- filled when system_correct = 0 (or = predicted when 1)
    action_fixed TEXT,                       -- one of the recommended actions, or free text
    comment TEXT,
    sdr_row_id INTEGER,                      -- simulated feedback: the SDR report it came from (full labels)
    round INTEGER                            -- simulated feedback: which feedback round
)"""
MIGRATIONS = {"sdr_row_id": "INTEGER", "round": "INTEGER"}  # columns added after the first version


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = cfg["project_root"] / "data" / "feedback" / "feedback.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or db_path())
    con.execute(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(feedback)")}
    for col, typ in MIGRATIONS.items():  # upgrade a database created by an older version
        if col not in have:
            con.execute(f"ALTER TABLE feedback ADD COLUMN {col} {typ}")
    return con


def add_many(records: list[dict], path: Path | None = None) -> None:
    if not records:
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recs = [{"ts": ts, "source": "app", **r} for r in records]
    cols = list(recs[0])
    with connect(path) as con:
        con.executemany(f"INSERT INTO feedback ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [[r[c] for c in cols] for r in recs])


def delete_source(source: str, path: Path | None = None) -> None:
    with connect(path) as con:
        con.execute("DELETE FROM feedback WHERE source = ?", (source,))


def add(record: dict, path: Path | None = None) -> int:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "app", **record}
    cols = ", ".join(rec)
    with connect(path) as con:
        cur = con.execute(f"INSERT INTO feedback ({cols}) VALUES ({', '.join('?' * len(rec))})", list(rec.values()))
        return int(cur.lastrowid)


def all_rows(path: Path | None = None):
    import pandas as pd
    with connect(path) as con:
        return pd.read_sql_query("SELECT * FROM feedback ORDER BY id", con)
