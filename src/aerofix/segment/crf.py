"""CRF segmenter (sklearn-crfsuite) with hand-crafted features.

A linear-chain CRF scores a whole tag sequence: per-token feature weights plus
transition weights between neighbouring tags (e.g. B-ACTION -> I-ACTION is common,
I-ACTION -> I-PROBLEM is rare). Viterbi decoding then picks the best sequence, so
the model can't produce an isolated PROBLEM token in the middle of an action.

Features describe the word and a +-2 word window. They encode what a human uses:
past-tense verbs, the action-verb and document-word lexicons, punctuation just
before (a new sentence), and position in the report (problems come first).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import sklearn_crfsuite

from aerofix.segment.common import ACTION_VERBS, DOC_WORDS, STEP_MARKER, from_bio, to_bio


def _word_feats(w: str, prefix: str) -> dict:
    return {
        f"{prefix}word": w,
        f"{prefix}suf3": w[-3:],
        f"{prefix}ed": w.endswith("ED"),
        f"{prefix}ing": w.endswith("ING"),
        f"{prefix}action_verb": w in ACTION_VERBS,
        f"{prefix}doc_word": w in DOC_WORDS,
        f"{prefix}has_digit": any(c.isdigit() for c in w),
        f"{prefix}punct": not any(c.isalnum() for c in w),
    }


def token_features(tokens: list[str], i: int, seen_action: bool) -> dict:
    w = tokens[i]
    f = {"bias": 1.0, "pos_frac": round(i / max(len(tokens) - 1, 1), 1),
         "step_marker": bool(STEP_MARKER.match(w)), "shape_ref": bool(w[:2].isdigit() and w.count("-") >= 1),
         "suf2": w[-2:], "pre3": w[:3], "len": min(len(w), 12)}
    f.update(_word_feats(w, ""))
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            f.update(_word_feats(tokens[j], f"{off:+d}:"))
        else:
            f[f"{off:+d}:edge"] = True
    # Has an action verb occurred anywhere before this token? (strong "we are in the action part" cue)
    f["after_action_verb"] = seen_action
    f["prev_sentence_end"] = i > 0 and tokens[i - 1] in {".", ";"}
    return f


def featurise(tokens: list[str]) -> list[dict]:
    feats, seen = [], False
    for i, t in enumerate(tokens):
        feats.append(token_features(tokens, i, seen))
        seen = seen or t in ACTION_VERBS  # running flag: O(n) instead of rescanning the prefix
    return feats


class CRFSegmenter:
    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 200):
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                                        all_possible_transitions=True)

    def fit(self, data: list[tuple[list[str], list[str]]]) -> "CRFSegmenter":
        self.crf.fit([featurise(t) for t, _ in data], [to_bio(lab) for _, lab in data])
        return self

    def predict(self, token_lists: list[list[str]]) -> list[list[str]]:
        return [from_bio(tags) for tags in self.crf.predict([featurise(t) for t in token_lists])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "CRFSegmenter":
        with open(path, "rb") as f:
            return pickle.load(f)
