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
    comment TEXT
)"""


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = cfg["project_root"] / "data" / "feedback" / "feedback.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or db_path())
    con.execute(SCHEMA)
    return con


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
