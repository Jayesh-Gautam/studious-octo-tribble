"""Shared pieces for problem/action segmentation.

Labels (one per token): PROBLEM, ACTION, REF, OTHER. Models are trained on BIO
tags (B-PROBLEM = first token of a problem span, I-PROBLEM = inside it), which
lets a sequence model mark where one span ends and the next begins.

Unlike the Phase 2 tokeniser, punctuation is KEPT as tokens: '.' and ';' are the
strongest clues for where a problem sentence ends.
"""
from __future__ import annotations

import re

LABELS = ["PROBLEM", "ACTION", "REF", "OTHER"]
BIO = ["O"] + [f"{p}-{lab}" for lab in LABELS for p in ("B", "I")]

SEG_TOKEN = re.compile(r"\[REF\]|\[\d+\]|[A-Z0-9]+(?:[&/'.\-][A-Z0-9]+)*|[^\sA-Z0-9]")
CHUNK_END = {".", ";", "<", "/", ":"}           # chunk ends after these
STEP_MARKER = re.compile(r"^\[\d+\]$")          # [01] [02]: step markers in recent reports

# Action triggers. Past-tense verbs dominate; shorthand forms come from the Phase 2 abbreviation data.
ACTION_VERBS = {
    "REPLACED", "REMOVED", "INSTALLED", "REINSTALLED", "REPAIRED", "SERVICED", "LUBRICATED",
    "RESEATED", "ADJUSTED", "TIGHTENED", "TORQUED", "CLEANED", "RESET", "SECURED", "REWORKED",
    "BLENDED", "DEFERRED", "TROUBLESHOT", "RIGGED", "SEALED", "RESEALED", "CHANGED", "FABRICATED",
    "RELEASED", "REPOSITIONED", "SWAPPED", "RESTORED", "TREATED", "R&R", "R/R", "RPLCD", "REPLCD",
    "RMVD", "RPRD", "RPR", "REPL", "REPLACE", "REMOVE", "REPAIR", "INSTALL", "SERVICE",
}
DOC_WORDS = {"SRM", "AMM", "MM", "M/M", "CMM", "MEL", "DWG", "IPC", "TSM", "FIM", "WDM", "SWPM",
             "NDT", "NDTM", "SSM", "DDG", "CDL", "EA", "EO", "ER"}


def seg_tokenize(text: str) -> list[str]:
    return SEG_TOKEN.findall(text.upper()) if text else []


def token_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in SEG_TOKEN.finditer(text.upper())] if text else []


def to_bio(labels: list[str]) -> list[str]:
    out, prev = [], None
    for lab in labels:
        out.append(("I-" if lab == prev else "B-") + lab)
        prev = lab
    return out


def from_bio(tags: list[str]) -> list[str]:
    """BIO -> plain labels. 'O' or a malformed tag inherits the previous label (OTHER at start)."""
    out, prev = [], "OTHER"
    for t in tags:
        lab = t.split("-", 1)[1] if "-" in t else prev
        out.append(lab)
        prev = lab
    return out


def spans(labels: list[str]) -> set[tuple[int, int, str]]:
    """Maximal runs of one label as (start, end_exclusive, label)."""
    out, start = set(), 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.add((start, i, labels[start]))
            start = i
    return out if labels else set()


def chunk_bounds(tokens: list[str]) -> list[int]:
    """Start index of each annotation chunk: after . ; < / : and before step markers,
    action verbs and document words, so most label changes fall on a chunk boundary."""
    starts = [0]
    for i in range(1, len(tokens)):
        prev, tok = tokens[i - 1], tokens[i]
        if prev in CHUNK_END or STEP_MARKER.match(tok) or tok in ACTION_VERBS or tok in DOC_WORDS:
            if starts[-1] != i:
                starts.append(i)
    return starts


def load_annotations(path) -> dict[str, list[tuple[list[str], list[str]]]]:
    """segments.jsonl -> {'train': [(tokens, labels), ...], 'dev': [...], 'test': [...]}.
    The last line per report wins (re-annotations); skipped reports are left out."""
    import json
    latest = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            latest[rec["row_id"]] = rec
    out = {"train": [], "dev": [], "test": []}
    for rec in latest.values():
        if not rec["skipped"]:
            out[rec["ann_split"]].append((rec["tokens"], rec["labels"]))
    return out


def join_tokens(tokens: list[str]) -> str:
    """Tokens -> readable text: no space before . , ; : ) and none after (."""
    text = " ".join(tokens)
    text = re.sub(r" ([.,;:)])", r"\1", text)
    return re.sub(r"\( ", "(", text)
