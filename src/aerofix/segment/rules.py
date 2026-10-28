"""Rule-based segmenter: the baseline every learned model must beat.

1. REF:     tokens inside a manual reference (Phase 1 regex) or a bare document
            word (SRM, AMM, DWG, MEL, ...).
2. OTHER:   step markers [01], control numbers like 2022-104311, '(OPEN)'.
3. Split:   everything before the first action verb is PROBLEM, everything after
            is ACTION. "R & R" written with spaces counts as an action verb.
"""
from __future__ import annotations

import re

from aerofix.preprocess.refs import find_refs
from aerofix.segment.common import ACTION_VERBS, DOC_WORDS, STEP_MARKER, seg_tokenize, token_offsets

CONTROL_NO = re.compile(r"^\d{4}-\d{4,}$")
OTHER_WORDS = {"OPEN", "CLOSED"}  # "(OPEN)" status tags


def first_action(tokens: list[str]) -> int | None:
    for i, t in enumerate(tokens):
        if t in ACTION_VERBS:
            return i
        if t == "R" and tokens[i + 1:i + 3] == ["&", "R"]:
            return i
    return None


def rule_labels(text: str) -> tuple[list[str], list[str]]:
    """Return (tokens, labels) for one report."""
    tokens, offsets = seg_tokenize(text), token_offsets(text)
    ref_spans = [(s, e) for s, e, *_ in find_refs(text.upper())]
    split = first_action(tokens)
    labels = []
    for i, (tok, (s, _)) in enumerate(zip(tokens, offsets)):
        if any(a <= s < b for a, b in ref_spans) or tok in DOC_WORDS:
            labels.append("REF")
        elif STEP_MARKER.match(tok) or CONTROL_NO.match(tok) or tok in OTHER_WORDS:
            labels.append("OTHER")
        elif split is not None and i >= split:
            labels.append("ACTION")
        else:
            labels.append("PROBLEM")
    return tokens, labels


def predict(token_lists: list[list[str]]) -> list[list[str]]:
    """Same interface as the learned models: tokens in, labels out."""
    return [rule_labels(" ".join(toks))[1] for toks in token_lists]
