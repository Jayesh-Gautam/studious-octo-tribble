"""Word n-gram language model (unigram / bigram / trigram), written from scratch.

Two smoothing methods, so the perplexity table has a baseline:

* Laplace (add-one): P(w | h) = (c(h, w) + 1) / (c(h) + |V|)
  Simple, but gives far too much mass to unseen events when |V| is large.

* Interpolated Kneser-Ney (Chen & Goodman 1998), fixed discount D:
    P(w | u, v) = max(c(uvw) - D, 0) / c(uv.)  +  D * T(uv.) / c(uv.) * P2(w | v)
    P2(w | v)   = max(N(.vw) - D, 0) / N(.v.)  +  D * T'(v.) / N(.v.) * P1(w)
    P1(w)       = N(.w) / N(..)          (continuation probability)
  where T(uv.) = number of distinct words seen after "u v", N(.vw) = number of
  distinct words seen before "v w". Idea: lower orders should model how many
  *different contexts* a word appears in. "FRANCISCO" is frequent but only after
  "SAN", so it should get a small continuation probability.
  If a context was never seen, its whole weight goes to the lower order.

Words seen fewer than ``unk_min_count`` times in training become <UNK>, so the
model has a probability for any input word. P1 is mixed with a tiny uniform term
so no probability is ever zero.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


class NGramLM:
    def __init__(self, order: int = 3, smoothing: str = "kn", discount: float = 0.75, unk_min_count: int = 2):
        assert order in (1, 2, 3) and smoothing in ("kn", "laplace")
        self.order, self.smoothing, self.d, self.unk_min = order, smoothing, discount, unk_min_count

    # ---------- training ----------
    def _pad(self, toks: list[str]) -> list[str]:
        toks = [t if t in self.vocab else UNK for t in toks]
        return [BOS] * (self.order - 1) + toks + [EOS]

    def fit(self, docs: Iterable[list[str]]) -> "NGramLM":
        docs = list(docs)
        raw = Counter(t for d in docs for t in d)
        self.vocab = {t for t, c in raw.items() if c >= self.unk_min} | {UNK, EOS}
        self.V = len(self.vocab)

        self.c1, self.c2, self.c3 = Counter(), Counter(), Counter()
        for d in docs:
            s = self._pad(d)
            for i in range(self.order - 1, len(s)):
                self.c1[s[i]] += 1
                if self.order >= 2:
                    self.c2[(s[i - 1], s[i])] += 1
                if self.order == 3:
                    self.c3[(s[i - 2], s[i - 1], s[i])] += 1
        self.n_tokens = sum(self.c1.values())

        # Context totals c(h.) and number of distinct followers T(h.)
        self.ctx2, self.fol2 = Counter(), Counter()
        for (v, _), c in self.c2.items():
            self.ctx2[v] += c
            self.fol2[v] += 1
        self.ctx3, self.fol3 = Counter(), Counter()
        for (u, v, _), c in self.c3.items():
            self.ctx3[(u, v)] += c
            self.fol3[(u, v)] += 1

        if self.smoothing == "kn":
            # Continuation counts: in how many distinct left contexts does an n-gram appear?
            self.cont1 = Counter(w for (_, w) in self.c2)                  # N(.w)
            self.cont1_total = len(self.c2)                                # N(..)
            self.cont2 = Counter((v, w) for (_, v, w) in self.c3)          # N(.vw)
            self.cont2_ctx, self.cont2_fol = Counter(), Counter()
            for (v, _), c in self.cont2.items():
                self.cont2_ctx[v] += c                                     # N(.v.)
                self.cont2_fol[v] += 1                                     # T'(v.)
        return self

    # ---------- probabilities ----------
    def _norm(self, w: str) -> str:
        return w if w in self.vocab else UNK

    def _p_kn1(self, w: str) -> float:
        if self.order == 1:
            p = self.c1[w] / self.n_tokens
        else:
            p = self.cont1[w] / self.cont1_total
        return 0.99 * p + 0.01 / self.V  # tiny uniform floor: never zero

    def _p_kn2(self, v: str, w: str, highest: bool) -> float:
        if highest:  # bigram model: raw counts at the top order
            c_vw, c_v, t_v = self.c2[(v, w)], self.ctx2[v], self.fol2[v]
        else:        # trigram model: continuation counts at the middle order
            c_vw, c_v, t_v = self.cont2[(v, w)], self.cont2_ctx[v], self.cont2_fol[v]
        if c_v == 0:
            return self._p_kn1(w)
        return max(c_vw - self.d, 0) / c_v + self.d * t_v / c_v * self._p_kn1(w)

    def prob(self, w: str, context: tuple[str, ...] = ()) -> float:
        """P(w | last order-1 words of context)."""
        w = self._norm(w)
        ctx = tuple(self._norm(c) if c != BOS else c for c in context)
        ctx = (BOS,) * (self.order - 1 - len(ctx)) + ctx if len(ctx) < self.order - 1 else ctx
        if self.smoothing == "laplace":
            if self.order == 1:
                return (self.c1[w] + 1) / (self.n_tokens + self.V)
            if self.order == 2:
                return (self.c2[(ctx[-1], w)] + 1) / (self.ctx2[ctx[-1]] + self.V)
            u, v = ctx[-2], ctx[-1]
            return (self.c3[(u, v, w)] + 1) / (self.ctx3[(u, v)] + self.V)
        if self.order == 1:
            return self._p_kn1(w)
        if self.order == 2:
            return self._p_kn2(ctx[-1], w, highest=True)
        u, v = ctx[-2], ctx[-1]
        c_uv = self.ctx3[(u, v)]
        lower = self._p_kn2(v, w, highest=False)
        if c_uv == 0:
            return lower
        return max(self.c3[(u, v, w)] - self.d, 0) / c_uv + self.d * self.fol3[(u, v)] / c_uv * lower

    def logprob(self, w: str, context: tuple[str, ...] = ()) -> float:
        return math.log(self.prob(w, context))

    def perplexity(self, docs: Iterable[list[str]]) -> float:
        """exp of the average negative log-probability per predicted token (incl. </s>)."""
        total, n = 0.0, 0
        for d in docs:
            s = [BOS] * (self.order - 1) + list(d) + [EOS]
            for i in range(self.order - 1, len(s)):
                total -= self.logprob(s[i], tuple(s[max(0, i - self.order + 1):i]))
                n += 1
        return math.exp(total / n)
