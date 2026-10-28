"""Manual-reference recommender.

Retrieval-based: collect the references cited in the retrieved similar cases.
  support(r) = similarity-weighted share of retrieved cases citing r
  global(r)  = share of ALL train reports citing r
  lift(r)    = support / global      (how much more often r appears here than usual)
  "count"        ranks by support only
  "specificity"  ranks by support * log(lift), keeping only lift > 1
SRM 51-40-02 (fastener installation) is cited by ~10% of all reports: it has high
support almost everywhere, but low lift. Specificity ranking pushes such generic
references down and the references particular to this kind of fault up.

Direct classifier: TF-IDF + one-vs-rest logistic regression (SGD) from problem text
to the most frequent references (multi-label), for comparison.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def global_rates(ref_lists) -> dict[str, float]:
    n = 0
    c: Counter = Counter()
    for refs in ref_lists:
        n += 1
        c.update(set(refs))
    return {r: k / n for r, k in c.items()}


def recommend(case_refs: list[list[str]], weights: np.ndarray, rates: dict[str, float], mode: str = "specificity",
              top: int = 5) -> list[tuple[str, float, float]]:
    """Returns (reference, score, support)."""
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = w.sum() or 1.0
    support: Counter = Counter()
    for refs, wi in zip(case_refs, w):
        for r in set(refs):
            support[r] += wi / total
    out = []
    for r, s in support.items():
        if mode == "count":
            out.append((r, s, s))
        else:
            lift = s / max(rates.get(r, 1e-6), 1e-6)
            if lift > 1:
                out.append((r, s * math.log(lift), s))
    return sorted(out, key=lambda x: -x[1])[:top]
