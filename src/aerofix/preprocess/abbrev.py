"""Abbreviation dictionary: mine candidates from train, apply a reviewed list.

Technicians write the same thing many ways: REPLACED / RPLCD / REPLCD / R&R. To a
bag-of-words model these are unrelated tokens. Mapping them to one form lets
TF-IDF, BM25 and Word2Vec pool their evidence.

Mining rules (train only), producing data/annotations/abbrev_candidates.csv:
  1. shape: 2-8 letters with no vowels, or containing & or / (RPLCD, CHK, P/N)
  2. contraction: the token's letters appear in order inside a much more frequent
     word starting with the same letter (RPLCD is a subsequence of REPLACED);
     that word is suggested as the expansion.
Candidates that are real English words (found in WordNet) are skipped by rule 2.
The final dictionary, data/annotations/abbrev.csv, is reviewed by hand; rows marked
ambiguous=yes (e.g. BS = body station or bolt shank?) are never expanded.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

VOWELS = set("AEIOU")


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _is_english(word: str) -> bool:
    from nltk.corpus import wordnet  # lazy: only mining needs WordNet
    return bool(wordnet.synsets(word.lower()))


def mine_candidates(counts: Counter, min_count: int = 20, ratio: int = 5, top: int = 400) -> list[dict]:
    """Return candidate rows {abbrev, count, rule, suggestion, suggestion_count}."""
    words = {w: c for w, c in counts.items() if w.isalpha() and len(w) >= 4}
    by_letter: dict[str, list[tuple[str, int]]] = {}
    for w, c in sorted(words.items(), key=lambda x: -x[1]):
        by_letter.setdefault(w[0], []).append((w, c))

    rows = []
    for tok, c in counts.most_common():
        if c < min_count or len(tok) < 2 or len(tok) > 8 or any(ch.isdigit() for ch in tok):
            continue
        letters = tok.replace("&", "").replace("/", "")
        if not letters.isalpha():
            continue
        shape = "&" in tok or "/" in tok or not (set(letters) & VOWELS)
        suggestion, s_count = "", 0
        if tok.isalpha() and len(tok) <= 6 and not _is_english(tok):
            for w, wc in by_letter.get(tok[0], []):
                if wc >= ratio * c and len(w) > len(tok) + 1 and _is_subsequence(tok, w):
                    suggestion, s_count = w, wc  # most frequent matching word wins
                    break
        if shape or suggestion:
            rows.append({"abbrev": tok, "count": c, "rule": "shape" if shape else "contraction",
                         "suggestion": suggestion, "suggestion_count": s_count})
        if len(rows) >= top:
            break
    return rows


def write_candidates(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_abbrev(path: Path) -> dict[str, list[str]]:
    """Reviewed dictionary -> {abbrev: expansion tokens}; ambiguous rows are skipped."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ambiguous"].strip().lower() != "yes" and row["expansion"].strip():
                out[row["abbrev"].strip().upper()] = row["expansion"].strip().upper().split()
    return out


def expand(tokens: list[str], table: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out.extend(table.get(t, [t]))
    return out
