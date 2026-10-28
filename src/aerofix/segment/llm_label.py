"""LLM-assisted annotation: python -m aerofix.segment.llm_label [--limit N]

Labels the annotation sample with the local Ollama model, chunk by chunk (the same
chunks the human tool shows), following docs/ANNOTATION_GUIDE.md. Output goes to
data/annotations/segments_llm.jsonl, separate from the human file
segments.jsonl; when both exist for a report, the HUMAN label wins (see
common.load_annotations). Reports already labelled by a human are not sent to the LLM.

Why chunk level: a 7B model is unreliable at counting individual words, but good at
judging a phrase ("REPLACED FUSE IN BATTERY PACK" -> ACTION). The reply must be
JSON with one required key per chunk ("1", "2", ...); Ollama's constrained
decoding enforces that schema, so long reports can't come back with a wrong
count. A failed reply is retried; after 3 failures the rule labels are used and
the record is marked ``fallback``.

Deterministic REF constraint (applied after the LLM): references are what the
Phase 1 regex is reliable at, so tokens inside a regex reference or a bare
document word (SRM, MM, DWG ...) are forced to REF, and an LLM "REF" chunk that
starts with an action verb ("REPAIRED IAW") becomes ACTION.

These labels are *silver* (machine-made); reports/segmentation.md states how many
labels came from the LLM vs a human.
"""
from __future__ import annotations

import argparse
import json
import time

import ollama
import pandas as pd

from aerofix.config import load_config
from aerofix.segment.common import ACTION_VERBS, LABELS, chunk_bounds, join_tokens
from aerofix.segment.rules import predict as rule_predict
from aerofix.utils import get_logger

log = get_logger(__name__)

SYSTEM = """You label aircraft maintenance reports (FAA Service Difficulty Reports).
Each report is split into numbered chunks. Give every chunk exactly one label:

PROBLEM: the fault and how/where it was found. e.g. "L1 DOOR DAMPER LEAKING", "DURING C-CHECK, FOUND CORROSION ON FRAME", "CREW REPORTED", "DEFECT SIZE: .187 INCH", "LOCATION: FRAME 500C AT STR 14R".
ACTION: what maintenance did, incl. checks and release. e.g. "REMOVED AND REPLACED DAMPER", "OPS CHECK GOOD", "MAINTENANCE" when followed by an action, "AIRCRAFT RELEASED FOR SERVICE", "DEFERRED PER", "TASK CARD COMPLETED", "INSPECTED WORK PERFORMED". Connector words like PER / IAW / REF belong to ACTION.
REF: the cited document itself. e.g. "AMM 52-10-00", "SRM 51-10-02 ITEM 9", "SRM", "MEL", "DWG 115A2714", "EA 97321".
OTHER: admin text. e.g. station or flight prefixes "PVD -", "LAX - FLT 36 -", step markers "[01]", control numbers "2022-104311", codes "( M )", "( W )", "(OPEN)", "GROUNDED - YES", tool calibration text.

Rules: "INSPECTED AND FOUND CRACK" is PROBLEM (finding the fault); "REPLACED CRACKED ANGLE" is ACTION (fixing it).
A chunk with only punctuation takes the label of the chunk before it.
Reply with JSON mapping every chunk number to its label, e.g. {"1": "PROBLEM", "2": "ACTION"}."""

# Worked examples (train reports / guide only; never test reports).
EXAMPLES = [
    (["PVD -", "ROWS 10, 11, 12L AISLE MARKER EMERGENCY LIGHTS INOPERATIVE.", "MAINTENANCE",
      "REPLACED FUSE IN BATTERY PACK M1479 AT STATION 598.", "( W )"],
     ["OTHER", "PROBLEM", "ACTION", "ACTION", "OTHER"]),
    (["CRACK REAR SPAR WEB SUPPORT AT RT WING", "2022-104311 /", "[01]", "REPLACED WEB SUPPORT IAW B737-700",
      "SRM 51-10-02 ITEM 9 AND FASTENER INSTALLED 51-40-02.;", "[02] TASK CARD COMPLETED"],
     ["PROBLEM", "OTHER", "OTHER", "ACTION", "REF", "ACTION"]),
]


def render(chunks: list[str]) -> str:
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))


def messages(chunks: list[str]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_chunks, ex_labels in EXAMPLES:
        msgs.append({"role": "user", "content": f"{len(ex_chunks)} chunks:\n{render(ex_chunks)}"})
        msgs.append({"role": "assistant",
                     "content": json.dumps({str(i + 1): lab for i, lab in enumerate(ex_labels)})})
    msgs.append({"role": "user", "content": f"{len(chunks)} chunks:\n{render(chunks)}"})
    return msgs


def schema(n: int) -> dict:
    """One required key per chunk, each restricted to the 4 labels."""
    keys = [str(i) for i in range(1, n + 1)]
    return {"type": "object", "properties": {k: {"type": "string", "enum": LABELS} for k in keys},
            "required": keys}


def apply_ref_constraint(tokens: list[str], labels: list[str], bounds: list[int], ends: list[int]) -> list[str]:
    rule = rule_predict([tokens])[0]  # rule REF = regex reference or bare document word
    out = list(labels)
    for s, e in zip(bounds, ends):
        if out[s] == "REF" and tokens[s] in ACTION_VERBS:
            out[s:e] = ["ACTION"] * (e - s)
    out = ["REF" if r == "REF" else lab for r, lab in zip(rule, out)]
    for i in range(1, len(tokens)):  # guide rule 4: punctuation takes the label of the word before it
        if not any(c.isalnum() for c in tokens[i]) and tokens[i] not in ("(", "["):
            out[i] = out[i - 1]
    return out


def label_report(client, model: str, tokens: list[str], retries: int = 3) -> tuple[list[str], dict]:
    bounds = chunk_bounds(tokens)
    ends = bounds[1:] + [len(tokens)]
    chunks = [join_tokens(tokens[s:e]) for s, e in zip(bounds, ends)]
    for attempt in range(1, retries + 1):
        try:
            r = client.chat(model=model, messages=messages(chunks), format=schema(len(chunks)),
                            options={"temperature": 0, "seed": 42})
            reply = json.loads(r["message"]["content"])
            chunk_labels = [reply.get(str(i + 1)) for i in range(len(chunks))]
            if all(lab in LABELS for lab in chunk_labels):
                labels = [lab for lab, s, e in zip(chunk_labels, bounds, ends) for _ in range(e - s)]
                return apply_ref_constraint(tokens, labels, bounds, ends), {"fallback": False, "attempts": attempt}
        except Exception as e:  # malformed JSON / connection hiccup -> retry
            log.warning("attempt %d failed: %s", attempt, e)
    return rule_predict([tokens])[0], {"fallback": True, "attempts": retries}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only label this many reports (for a trial run)")
    args = ap.parse_args()
    cfg = load_config()
    ann = cfg["paths"]["annotations"]
    sample = pd.read_parquet(ann / "seg_sample.parquet")
    human = set()
    if (ann / "segments.jsonl").exists():
        human = {json.loads(line)["row_id"] for line in open(ann / "segments.jsonl", encoding="utf-8") if line.strip()}
    out_path = ann / "segments_llm.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["row_id"] for line in open(out_path, encoding="utf-8") if line.strip()}
    todo = [r for r in sample.itertuples() if r.row_id not in human and r.row_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    model = cfg["llm"]["model"]
    client = ollama.Client(host=cfg["llm"]["host"])
    log.info("labelling %d reports with %s (%d human-labelled, %d already done)", len(todo), model, len(human), len(done))

    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            tokens = list(r.tokens)
            t = time.perf_counter()
            labels, meta = label_report(client, model, tokens)
            f.write(json.dumps({"row_id": int(r.row_id), "ann_split": r.ann_split, "tokens": tokens, "labels": labels,
                                "skipped": False, "annotator": model, "seconds": round(time.perf_counter() - t, 1),
                                **meta}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                log.info("%d / %d (%.0f s elapsed)", i, len(todo), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
