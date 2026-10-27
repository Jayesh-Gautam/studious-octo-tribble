"""Spelling correction for maintenance text: edit distance + n-gram LM (noisy channel).

Dictionary: train words seen at least ``dict_min_count`` times are assumed correct.
Any other alphabetic token of length >= ``min_len`` is a correction candidate.

1. Candidate generation (SymSpell idea): pre-compute every string obtained by
   deleting up to 2 letters from each dictionary word. A typo and its correct word
   share a delete-string if they are within 2 edits, so lookup is a dict hit
   instead of comparing against all 12k words.
2. Exact Damerau-Levenshtein distance (insert, delete, substitute, swap adjacent
   letters) filters candidates to distance <= max_edit.
3. Ranking, two modes:
     "edit": smallest distance, ties -> most frequent word (Norvig-style baseline)
     "lm":   noisy channel  score(w) = log P_LM(w | 2 left words)
                                     + log P_LM(next word | left word, w)
                                     - edit_penalty * distance
   The LM term lets context decide, e.g. "CORRISION FOUND" vs "CORRECTION MADE".

Never touched: tokens with digits or symbols (part numbers, P/N, [REF]), tokens
shorter than min_len, dictionary words, abbreviation-dictionary entries, and real
English words found in WordNet. The last rule was added after the first run
"corrected" rare but valid words (LOADABLE -> READABLE, DECELERATED -> ACCELERATED).
Limitation: real-word errors ("FROM" typed as "FORM") are not detected, because
both are valid words.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations

from aerofix.preprocess.ngram_lm import NGramLM


def damerau_levenshtein(a: str, b: str) -> int:
    """Optimal-string-alignment distance (adjacent transposition counts as 1 edit)."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


@lru_cache(maxsize=200_000)
def is_english(word: str) -> bool:
    """True if WordNet knows the word (inflections included: LEAKS, CRACKED)."""
    from nltk.corpus import wordnet
    return bool(wordnet.synsets(word.lower()))


def _deletes(word: str, max_edit: int) -> set[str]:
    out = {word}
    for k in range(1, min(max_edit, len(word) - 1) + 1):
        for idx in combinations(range(len(word)), k):
            out.add("".join(ch for i, ch in enumerate(word) if i not in idx))
    return out


class SpellCorrector:
    def __init__(self, counts: Counter, lm: NGramLM | None = None, protected: set[str] = frozenset(),
                 dict_min_count: int = 5, max_edit: int = 2, min_len: int = 4, edit_penalty: float = 3.0):
        self.freq = {w: c for w, c in counts.items() if w.isalpha() and c >= dict_min_count}
        self.lm, self.protected = lm, set(protected)
        self.max_edit, self.min_len, self.penalty = max_edit, min_len, edit_penalty
        self.index: dict[str, list[str]] = {}
        for w in self.freq:
            for d in _deletes(w, max_edit):
                self.index.setdefault(d, []).append(w)
        self._candidates = lru_cache(maxsize=200_000)(self._candidates_uncached)

    # The lru_cache wrapper can't be pickled: drop it on save, rebuild on load.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_candidates", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._candidates = lru_cache(maxsize=200_000)(self._candidates_uncached)

    def needs_correction(self, tok: str) -> bool:
        return (tok.isalpha() and len(tok) >= self.min_len and tok not in self.freq
                and tok not in self.protected and not is_english(tok))

    def _candidates_uncached(self, tok: str) -> tuple[tuple[str, int], ...]:
        seen = set()
        for d in _deletes(tok, self.max_edit):
            seen.update(self.index.get(d, ()))
        out = [(w, damerau_levenshtein(tok, w)) for w in seen]
        return tuple((w, dist) for w, dist in out if dist <= self.max_edit)

    def best(self, tok: str, left: tuple[str, ...] = (), right: str | None = None, mode: str = "lm") -> str:
        """Correction for one token given up to 2 left words and 1 right word."""
        if not self.needs_correction(tok):
            return tok
        cands = self._candidates(tok)
        if not cands:
            return tok
        if mode == "edit" or self.lm is None:
            return min(cands, key=lambda x: (x[1], -self.freq[x[0]]))[0]

        def score(wd):
            w, dist = wd
            s = self.lm.logprob(w, left[-2:]) - self.penalty * dist
            if right is not None:
                s += self.lm.logprob(right, (left[-1] if left else "<s>", w))
            return s
        return max(cands, key=score)[0]

    def correct(self, tokens: list[str], mode: str = "lm") -> list[str]:
        """Correct left to right, so later decisions see already-corrected context."""
        out: list[str] = []
        for i, t in enumerate(tokens):
            right = tokens[i + 1] if i + 1 < len(tokens) else None
            out.append(self.best(t, tuple(out[-2:]), right, mode))
        return out
