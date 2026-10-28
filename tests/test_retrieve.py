import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from aerofix.retrieve.common import gains, metrics_for, rrf
from aerofix.retrieve.index import BM25, DenseIndex, normalise_rows
from aerofix.retrieve.train import unique_group_batches

DOCS = ["DOOR DAMPER LEAKING", "EMERGENCY LIGHT BATTERY INOP", "CRACK ON FRAME 16", "DOOR SEAL TORN",
        "CORROSION ON FRAME", "EMERGENCY LIGHT LENS BROKEN"]


def test_bm25_matches_rank_bm25():
    ours = BM25(k1=1.5, b=0.75).fit(DOCS)
    ref = BM25Okapi([d.split() for d in DOCS], k1=1.5, b=0.75)
    for q in ["DOOR LEAKING", "EMERGENCY LIGHT", "FRAME"]:
        a, b = ours.scores(q), ref.get_scores(q.split())
        # rank-bm25 uses idf = ln((N - df + .5) / (df + .5)) (can be negative) while ours uses ln(1 + ...);
        # the ORDER of documents with positive scores must agree.
        top_a = [i for i in np.argsort(-a) if a[i] > 0]
        top_b = [i for i in np.argsort(-b) if b[i] > 0]
        assert top_a[:2] == top_b[:2] or len(top_b) < 2


def test_bm25_search_and_mask():
    bm = BM25().fit(DOCS)
    assert bm.search("EMERGENCY LIGHT BATTERY", 2).tolist()[0] == 1
    mask = np.array([True, False, True, True, True, True])
    assert 1 not in bm.search("EMERGENCY LIGHT BATTERY", 3, mask=mask).tolist()


def test_dense_index_respects_family():
    v = normalise_rows(np.eye(4, dtype=np.float32) + 0.01)
    idx = DenseIndex(v, np.array(["A320", "A320", "B737", "B737"]))
    res = idx.search(v[[0, 2]], np.array(["A320", "B737"]), 2)
    assert set(res[0]) <= {0, 1} and set(res[1]) <= {2, 3} and res[0][0] == 0


def test_rrf_rewards_agreement():
    out = rrf([np.array([1, 2, 3]), np.array([3, 1, 4])], top=4)
    assert out[0] in (1, 3) and set(out) == {1, 2, 3, 4}


def test_gains_and_metrics():
    q = pd.Series({"ata4": "5210", "condition": "LEAKING"})
    c = pd.DataFrame({"ata4": ["5210", "5210", "3350"], "condition": ["CRACKED", "LEAKING", "LEAKING"]})
    g = gains(q, c)
    assert g.tolist() == [1, 2, 0]
    m = metrics_for(g)
    assert m["Hit@5"] == 1.0 and m["MRR@10"] == 0.5 and 0 < m["NDCG@5"] < 1


def test_unique_group_batches():
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    for b in unique_group_batches(groups, 2, np.random.default_rng(0)):
        assert len({groups[i] for i in b}) == len(b)
