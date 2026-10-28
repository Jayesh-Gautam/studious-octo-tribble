"""Shared retrieval pieces: data, proxy relevance, metrics, rank fusion.

Proxy relevance: a past report is relevant to a query if it is from the same
aircraft family AND has the same JASC code (ata4) AND the same part condition.
That is "the same kind of fault on the same kind of part". There are usually many
relevant reports per query, so recall in the classical sense (share of ALL relevant
documents found) is meaningless here; we report:
    Hit@k     share of queries with at least one relevant report in the top k
    P@5       share of the top 5 that are relevant
    MRR@10    1 / rank of the first relevant report (0 if none in the top 10)
    NDCG@5    ranking quality with graded gain: 2 = same ata4 + condition,
              1 = same ata4 only, 0 = otherwise (so "right part, wrong condition"
              earns partial credit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aerofix.config import load_config


def load_corpus(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    proc = cfg["paths"]["processed"]
    df = pd.read_parquet(proc / "sdr_a320_b737.parquet",
                         columns=["row_id", "split", "family", "ata2", "ata4", "PartCondition", "text"])
    df = df.merge(pd.read_parquet(proc / "sdr_segments.parquet", columns=["row_id", "problem_masked", "action_text"]),
                  on="row_id")
    df = df.merge(pd.read_parquet(proc / "sdr_problem_norm.parquet"), on="row_id")
    df["condition"] = df.PartCondition.fillna("").str.strip().str.upper()
    return df.reset_index(drop=True)


def gains(query: pd.Series, cands: pd.DataFrame) -> np.ndarray:
    same4 = (cands.ata4.values == query.ata4)
    both = same4 & (cands.condition.values == query.condition)
    return np.where(both, 2, np.where(same4, 1, 0))


def metrics_for(g: np.ndarray) -> dict:
    """g = graded gains of the ranked list for ONE query (top 10 at least)."""
    rel = g == 2
    first = np.argmax(rel[:10]) if rel[:10].any() else None
    dcg = sum((2 ** g[i] - 1) / np.log2(i + 2) for i in range(min(5, len(g))))
    ideal = sorted(g, reverse=True)  # approximation: ideal ordering of the retrieved list
    idcg = sum((2 ** ideal[i] - 1) / np.log2(i + 2) for i in range(min(5, len(ideal))))
    return {"Hit@5": float(rel[:5].any()), "Hit@10": float(rel[:10].any()), "P@5": float(rel[:5].mean()),
            "MRR@10": 0.0 if first is None else 1.0 / (first + 1),
            "NDCG@5": 0.0 if idcg == 0 else dcg / idcg}


def rrf(rankings: list[np.ndarray], k: int = 60, top: int = 50) -> np.ndarray:
    """Reciprocal rank fusion: score(d) = sum over rankers of 1 / (k + rank). Needs no score
    calibration between BM25 (unbounded) and cosine (-1..1), which is why it is the usual hybrid."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, d in enumerate(r):
            score[int(d)] = score.get(int(d), 0.0) + 1.0 / (k + rank + 1)
    return np.array(sorted(score, key=score.get, reverse=True)[:top])
