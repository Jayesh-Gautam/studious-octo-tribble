"""Normalise the Phase 3 problem text: python -m aerofix.classify.prepare

Applies the Phase 2 normaliser (fitted on train) to ``problem_masked`` and writes
data/processed/sdr_problem_norm.parquet with problem_norm (abbreviations +
spelling), problem_stem and problem_lemma. Used by the normalisation ablation and
by the BiLSTM / Word2Vec model.
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
    seg = pd.read_parquet(cfg["paths"]["processed"] / "sdr_segments.parquet", columns=["row_id", "problem_masked"])
    norm = Normaliser.load(cfg["paths"]["models"] / "normaliser.pkl")
    t = time.perf_counter()
    base = [norm.tokens(x, morph="none") for x in seg.problem_masked]
    log.info("abbrev + spelling: %.0f s", time.perf_counter() - t)
    out = pd.DataFrame({"row_id": seg.row_id, "problem_norm": [" ".join(b) for b in base],
                        "problem_stem": [" ".join(norm.stem(b)) for b in base]})
    t = time.perf_counter()
    out["problem_lemma"] = [" ".join(norm.lemmatise(b)) for b in base]
    log.info("lemmatisation: %.0f s", time.perf_counter() - t)
    path = cfg["paths"]["processed"] / "sdr_problem_norm.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
