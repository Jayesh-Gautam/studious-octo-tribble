"""The retriever used by the recommenders and the app (method chosen on val in Phase 5).

Hybrid: BM25 on lemmatised problem text + fine-tuned MiniLM, merged with reciprocal
rank fusion, then the top 50 re-ranked by the cross-encoder. Returned similarity is
the fine-tuned cosine between query and case (bounded 0..1-ish, so it can weight
votes in the recommenders; cross-encoder logits are only used for ordering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from aerofix.config import load_config
from aerofix.retrieve.common import rrf
from aerofix.retrieve.index import BM25, DenseIndex
from aerofix.retrieve.train import CrossEncoder, encode


class Retriever:
    def __init__(self, docs: pd.DataFrame, vectors: np.ndarray, cfg: dict | None = None, rerank: bool = True):
        """docs: index reports (needs family, problem_masked, problem_lemma); vectors: their fine-tuned embeddings."""
        cfg = cfg or load_config()
        rdir = cfg["paths"]["models"] / "retrieve"
        self.docs = docs.reset_index(drop=True)
        self.vectors = vectors.astype(np.float32)
        self.family = self.docs.family.values
        self.bm25 = BM25().fit(list(self.docs.problem_lemma.fillna("")))
        self.dense = DenseIndex(self.vectors, self.family)
        self.encoder = SentenceTransformer(str(rdir / "biencoder"))
        self.cross = CrossEncoder(str(rdir / "crossencoder")) if rerank else None

    def embed(self, texts: list[str]) -> np.ndarray:
        return encode(self.encoder, texts)

    def search(self, problem: str, problem_lemma: str, family: str, k: int = 20, qvec: np.ndarray | None = None,
               pool: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc positions, cosine similarities) of the top k cases, same family only."""
        qvec = self.embed([problem])[0] if qvec is None else qvec
        lex = self.bm25.search(problem_lemma, pool, mask=self.family == family)
        sem = self.dense.search(qvec[None], np.array([family]), pool)[0]
        cand = rrf([lex, sem], top=pool)
        if self.cross is not None:
            s = self.cross.score(problem, list(self.docs.problem_masked.values[cand]))
            cand = cand[np.argsort(-s)]
        top = cand[:k]
        return top, self.vectors[top] @ qvec
