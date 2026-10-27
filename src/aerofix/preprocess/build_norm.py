"""Fit the normaliser on train and normalise the whole corpus.

    python -m aerofix.preprocess.build_norm

Writes models/normaliser.pkl and data/processed/sdr_norm.parquet with columns
row_id, text_norm (abbrev + spelling), text_stem, text_lemma, all built from
``text_masked`` so no manual reference numbers reach the models.
"""
from __future__ import annotations

import time

import pandas as pd

from aerofix.config import load_config
from aerofix.preprocess.normalise import Normaliser
from aerofix.utils import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet",
                         columns=["row_id", "text_masked", "split"])

    t = time.perf_counter()
    norm = Normaliser.fit(df.text_masked[df.split == "train"], cfg)
    log.info("fitted normaliser in %.0f s: %d abbreviations, %d dictionary words, LM vocab %d",
             time.perf_counter() - t, len(norm.abbrev), len(norm.spell.freq), norm.lm.V)

    t = time.perf_counter()
    base = [norm.tokens(x, morph="none") for x in df.text_masked]
    log.info("abbrev + spelling done in %.0f s", time.perf_counter() - t)
    t = time.perf_counter()
    stem = [" ".join(norm.stem(toks)) for toks in base]
    log.info("stemming done in %.0f s", time.perf_counter() - t)
    t = time.perf_counter()
    lemma = [" ".join(norm.lemmatise(toks)) for toks in base]
    log.info("lemmatisation done in %.0f s", time.perf_counter() - t)

    out = pd.DataFrame({"row_id": df.row_id, "text_norm": [" ".join(x) for x in base],
                        "text_stem": stem, "text_lemma": lemma})
    path = cfg["paths"]["processed"] / "sdr_norm.parquet"
    out.to_parquet(path, index=False)
    norm.save(cfg["paths"]["models"] / "normaliser.pkl")
    log.info("wrote %s and models/normaliser.pkl", path)


if __name__ == "__main__":
    main()
