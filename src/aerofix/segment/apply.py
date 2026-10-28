"""Segment the whole corpus with the best model: python -m aerofix.segment.apply

Reads models/segment/best.json (chosen on dev by segment.evaluate) and writes
data/processed/sdr_segments.parquet:
    row_id, problem_text, action_text, ref_text, problem_masked, has_action
``problem_masked`` (problem text with any leftover reference numbers masked) is
the input for Phase 4 classification: it is what a technician knows before fixing.
If a report gets no PROBLEM tokens (e.g. it starts with "REPLACED ..."), the
whole masked report is used as problem text and the row is counted in the log.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from aerofix.config import load_config
from aerofix.preprocess.refs import mask_refs
from aerofix.segment import rules
from aerofix.segment.common import join_tokens, seg_tokenize
from aerofix.utils import get_logger

log = get_logger(__name__)


def load_best(mdir):
    best = json.loads((mdir / "best.json").read_text())["best"]
    if best == "CRF":
        from aerofix.segment.crf import CRFSegmenter
        return best, CRFSegmenter.load(mdir / "crf.pkl")
    if best == "DistilBERT":
        from aerofix.segment.bert import BertSegmenter
        return best, BertSegmenter.load(mdir / "bert")
    return best, rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=20000, help="reports per prediction batch")
    args = ap.parse_args()
    cfg = load_config()
    mdir = cfg["paths"]["models"] / "segment"
    name, model = load_best(mdir)
    df = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet", columns=["row_id", "text", "text_masked"])
    token = cfg["preprocess"]["mask_token"]
    log.info("segmenting %d reports with %s", len(df), name)

    out, t = [], time.perf_counter()
    for start in range(0, len(df), args.chunk):
        part = df.iloc[start:start + args.chunk]
        toks = [seg_tokenize(x) for x in part.text]
        labels = model.predict(toks)
        for rid, tk, lab, masked in zip(part.row_id, toks, labels, part.text_masked):
            pick = lambda name: join_tokens([w for w, lb in zip(tk, lab) if lb == name])
            problem = pick("PROBLEM")
            out.append({"row_id": rid, "problem_text": problem, "action_text": pick("ACTION"),
                        "ref_text": pick("REF"),
                        "problem_masked": mask_refs(problem, token) if problem else masked,
                        "problem_fallback": not problem, "has_action": "ACTION" in lab})
        log.info("%d / %d done (%.0f s)", min(start + args.chunk, len(df)), len(df), time.perf_counter() - t)

    res = pd.DataFrame(out)
    path = cfg["paths"]["processed"] / "sdr_segments.parquet"
    res.to_parquet(path, index=False)
    log.info("wrote %s | segmenter %s | has_action %.1f%% | problem fallback %d rows | median problem words %d",
             path, name, 100 * res.has_action.mean(), int(res.problem_fallback.sum()),
             int(res.problem_text.str.split().str.len().median()))


if __name__ == "__main__":
    main()
