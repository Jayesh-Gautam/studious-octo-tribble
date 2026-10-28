"""Sparse BM25 and dense (FAISS) indexes.

BM25 (Okapi), written on scipy sparse matrices so a query against ~400k reports
takes milliseconds (rank-bm25 loops in Python and needs ~0.5 s):
    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
k1 limits how much repeating a word helps (saturation); b normalises for length,
so a long report doesn't win just by containing more words.

Dense: vectors are L2-normalised, so inner product = cosine similarity; FAISS
IndexFlatIP does exact search (no approximation needed at this size).
"""
from __future__ import annotations

import faiss
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> "BM25":
        self.vec = CountVectorizer(token_pattern=r"\S+", lowercase=False)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        # Pre-compute the per-(term, doc) BM25 weight once; a query is then a sparse dot product.
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.w = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])  # each distinct query term counts once
        if not q.nnz:
            return np.zeros(self.w.shape[0], dtype=np.float32)
        # Summing the query terms' columns touches only their posting lists (fast on CSC).
        return np.asarray(self.w[:, q.indices].sum(1)).ravel()

    def search(self, query: str, top: int, mask: np.ndarray | None = None) -> np.ndarray:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        top = min(top, len(s))
        idx = np.argpartition(-s, top - 1)[:top]
        return idx[np.argsort(-s[idx])]


class DenseIndex:
    """One exact inner-product index per aircraft family (search never crosses families)."""

    def __init__(self, vectors: np.ndarray, groups: np.ndarray):
        self.indexes, self.ids = {}, {}
        for g in np.unique(groups):
            sel = np.where(groups == g)[0]
            ix = faiss.IndexFlatIP(vectors.shape[1])
            ix.add(np.ascontiguousarray(vectors[sel], dtype=np.float32))
            self.indexes[g], self.ids[g] = ix, sel

    def search(self, qvecs: np.ndarray, qgroups: np.ndarray, top: int) -> list[np.ndarray]:
        out = [None] * len(qvecs)
        for g in np.unique(qgroups):
            qi = np.where(qgroups == g)[0]
            _, nn = self.indexes[g].search(np.ascontiguousarray(qvecs[qi], dtype=np.float32), top)
            for j, row in zip(qi, nn):
                out[j] = self.ids[g][row[row >= 0]]
        return out


def normalise_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
