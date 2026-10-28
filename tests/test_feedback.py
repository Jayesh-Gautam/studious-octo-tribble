import sqlite3

import pandas as pd

from aerofix.feedback import store


def test_add_read_delete(tmp_path):
    db = tmp_path / "fb.sqlite"
    rid = store.add({"text": "DOOR LEAK", "family": "A320", "predicted_ata2": "52", "system_correct": 1,
                     "correct_ata2": "52"}, db)
    store.add_many([{"source": "simulated", "round": 1, "text": "X", "correct_ata2": "33", "sdr_row_id": 7}], db)
    rows = store.all_rows(db)
    assert rid == 1 and len(rows) == 2 and rows.iloc[1]["round"] == 1 and rows.iloc[1].sdr_row_id == 7
    store.delete_source("simulated", db)
    assert store.all_rows(db).source.tolist() == ["app"]


def test_old_database_is_migrated(tmp_path):
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
                "source TEXT NOT NULL DEFAULT 'app', text TEXT NOT NULL)")
    con.commit()
    con.close()
    store.add_many([{"text": "Y", "sdr_row_id": 3, "round": 2}], db)  # needs the new columns
    assert set(pd.Series(store.all_rows(db).columns)) >= {"sdr_row_id", "round"}
