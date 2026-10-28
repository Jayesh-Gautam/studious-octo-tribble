"""Canonical corrective actions: "REMOVED AND REPLACED LH AFT EMERGENCY LIGHT POWER PACK
M1673 IAW AMM" -> "REPLACE POWER PACK".

1. Rules (on the Phase 3 action text, abbreviations expanded so RPLCD / R&R become
   REPLACED / REMOVED AND REPLACED):
   - find a fix verb and map it to a base form (REPLACED, CHANGED, SWAPPED -> REPLACE)
   - "REMOVED AND REPLACED X" counts once, as REPLACE X
   - object = the words after the verb up to a preposition, conjunction, punctuation
     or next verb; drop articles/adjectives (THE, NEW, SERVICEABLE), side/position
     words (LH, AFT, FORWARD) and anything with a digit (part numbers, stations)
   - head = the last 2 remaining words ("EMERGENCY LIGHT POWER PACK" -> "POWER PACK")
   Checks and sign-offs (OPS CHECK GOOD, INSPECTED WORK) are not recommended fixes
   and are ignored: they follow almost every repair and would top every list.
2. Clustering merges spelling/synonym variants of frequent objects: object strings
   are embedded with MiniLM and grouped by agglomerative clustering (average linkage,
   cosine distance < threshold); each object maps to its cluster's most frequent
   member (e.g. "BATTERY PACKS" -> "BATTERY PACK").
"""
from __future__ import annotations

from collections import Counter

import numpy as np

VERBS = {
    "REPLACED": "REPLACE", "REPLACE": "REPLACE", "CHANGED": "REPLACE", "SWAPPED": "REPLACE",
    "REMOVED": "REMOVE", "INSTALLED": "INSTALL", "REINSTALLED": "INSTALL", "REPAIRED": "REPAIR",
    "FABRICATED": "REPAIR", "SERVICED": "SERVICE", "LUBRICATED": "LUBRICATE", "ADJUSTED": "ADJUST",
    "RIGGED": "ADJUST", "RESEATED": "RESEAT", "SECURED": "SECURE", "RESECURED": "SECURE",
    "TIGHTENED": "SECURE", "TORQUED": "SECURE", "CLEANED": "CLEAN", "RESET": "RESET", "REWORKED": "REWORK",
    "BLENDED": "BLEND", "DEFERRED": "DEFER", "TREATED": "TREAT", "SEALED": "SEAL", "RESEALED": "SEAL",
    "TROUBLESHOT": "TROUBLESHOOT", "RESTORED": "RESTORE", "REPOSITIONED": "ADJUST", "RE-INSTALLED": "INSTALL",
}
STOP = {"IN", "ON", "AT", "PER", "IAW", "AND", "WITH", "FROM", "TO", "OF", "FOR", "BY", "AS", "REF", "USING",
        "DUE", "THEN", "WHICH", "THAT", "NOT", "NO", "BETWEEN", "INTO", "UNDER", "OVER", "WITHIN", "OR", "[REF]",
        "ABOVE", "BELOW", "NEAR", "LOCATED", "AFTER", "BEFORE", "DURING", "WHILE", "WAS", "WERE", "IS", "ARE", "AS"}
SKIP = {"THE", "A", "AN", "NEW", "SERVICEABLE", "S/V", "DAMAGED", "DEFECTIVE", "AFFECTED", "SUBJECT", "ALL", "BOTH",
        "LEFT", "RIGHT", "LH", "RH", "FORWARD", "AFT", "FWD", "UPPER", "LOWER", "INBOARD", "OUTBOARD", "CENTER",
        "NUMBER", "NR", "NO.", "ITEM", "ONE", "TWO", "SEVERAL", "EXISTING", "OLD", "CRACKED", "CORRODED", "BROKEN",
        "WORN", "LOOSE", "MISSING", "FAILED", "INOPERATIVE", "FAULTY", "LEAKING"}


def action_tokens(text: str, abbrev: dict[str, list[str]]) -> list[str]:
    """Tokens WITH punctuation (a '.' ends an object) and abbreviations expanded (RPLCD -> REPLACED)."""
    from aerofix.preprocess.abbrev import expand
    from aerofix.segment.common import seg_tokenize
    return expand(seg_tokenize(text or ""), abbrev)


def extract_actions(tokens: list[str]) -> list[str]:
    """Canonical actions ("VERB OBJECT" or "VERB") in order of appearance, without duplicates."""
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i:i + 3] == ["R", "&", "R"]:  # "R & R" written with spaces
            tokens = tokens[:i] + ["REPLACED"] + tokens[i + 3:]
        verb = VERBS.get(tokens[i])
        if verb is None:
            i += 1
            continue
        # "REMOVED AND REPLACED X" -> one REPLACE action
        if verb == "REMOVE" and i + 2 < len(tokens) and tokens[i + 1] == "AND" and VERBS.get(tokens[i + 2]) == "REPLACE":
            i += 2
            continue
        obj = []
        j = i + 1
        while (j < len(tokens) and tokens[j] not in STOP and tokens[j] not in VERBS and len(obj) < 8
               and any(c.isalnum() for c in tokens[j])):  # punctuation ends the object
            t = tokens[j]
            if t.isalpha() and t not in SKIP:
                obj.append(t)
            j += 1
        action = verb + (" " + " ".join(obj[-2:]) if obj else "")
        if action not in out:
            out.append(action)
        i = j
    return out


def split_action(a: str) -> tuple[str, str]:
    verb, _, obj = a.partition(" ")
    return verb, obj


def cluster_objects(objects: Counter, encoder_fn, min_count: int = 20, threshold: float = 0.15) -> dict[str, str]:
    """Map object strings to a cluster representative (most frequent member). Objects seen fewer
    than min_count times keep their own name (too rare to cluster reliably)."""
    from sklearn.cluster import AgglomerativeClustering
    frequent = [o for o, c in objects.most_common() if c >= min_count and o]
    if len(frequent) < 2:
        return {}
    vecs = encoder_fn([o.lower() for o in frequent])
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                     linkage="average").fit_predict(vecs)
    rep: dict[int, str] = {}
    for o, lab in zip(frequent, labels):  # frequent is sorted by count: first member = most frequent
        rep.setdefault(lab, o)
    return {o: rep[lab] for o, lab in zip(frequent, labels) if rep[lab] != o}


def canonicalise(actions: list[str], mapping: dict[str, str]) -> list[str]:
    out = []
    for a in actions:
        v, o = split_action(a)
        c = v + (" " + mapping.get(o, o) if o else "")
        if c not in out:
            out.append(c)
    return out


def aggregate(case_actions: list[list[str]], weights: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
    """Similarity-weighted vote over retrieved cases. Returns (action, share of total weight)."""
    score: Counter = Counter()
    for acts, w in zip(case_actions, weights):
        for a in acts:
            score[a] += float(max(w, 0.0))
    total = sum(max(float(w), 0.0) for w in weights) or 1.0
    return [(a, s / total) for a, s in score.most_common(top)]
