"""Train and compare segmenters: python -m aerofix.segment.evaluate

Rules vs CRF vs DistilBERT on the hand-annotated reports (train / dev / test).
Model choices (CRF c1/c2, BERT epochs) use dev only; the 60 test reports, which
were annotated without rule pre-fill, are scored once at the end.

Metrics
  token F1: per label over all tokens, plus the macro average of the 4 labels
  span F1:  a predicted span counts only if start, end and label all match a gold
            span exactly (strict, like CoNLL NER evaluation); micro over all spans
Writes reports/segmentation.md and models/segment/{crf.pkl, bert/, best.json}.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd

from aerofix.config import load_config
from aerofix.segment import rules
from aerofix.segment.common import LABELS, load_annotations, spans
from aerofix.utils import get_logger, md_table, set_seed

log = get_logger(__name__)


def token_scores(gold: list[list[str]], pred: list[list[str]]) -> dict:
    g = [x for doc in gold for x in doc]
    p = [x for doc in pred for x in doc]
    out = {}
    for lab in LABELS:
        tp = sum(a == lab and b == lab for a, b in zip(g, p))
        fp = sum(a != lab and b == lab for a, b in zip(g, p))
        fn = sum(a == lab and b != lab for a, b in zip(g, p))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out["macro"] = sum(out[lab] for lab in LABELS) / len(LABELS)
    return out


def token_macro(gold, pred) -> float:
    return token_scores(gold, pred)["macro"]


def span_scores(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tp = n_pred = n_gold = 0
    per = {lab: [0, 0, 0] for lab in LABELS}  # tp, pred, gold
    for g, p in zip(gold, pred):
        gs, ps = spans(g), spans(p)
        tp += len(gs & ps)
        n_pred += len(ps)
        n_gold += len(gs)
        for lab in LABELS:
            gl = {s for s in gs if s[2] == lab}
            pl = {s for s in ps if s[2] == lab}
            per[lab][0] += len(gl & pl)
            per[lab][1] += len(pl)
            per[lab][2] += len(gl)
    f = lambda t, np_, ng: 2 * t / (np_ + ng) if np_ + ng else 0.0
    out = {"micro": f(tp, n_pred, n_gold)}
    out.update({lab: f(*per[lab]) for lab in LABELS})
    return out


def score_row(name: str, split: str, gold, pred) -> dict:
    t, s = token_scores(gold, pred), span_scores(gold, pred)
    row = {"method": name, "split": split, "token_macro_F1": round(t["macro"], 4)}
    row.update({f"tok_{lab}": round(t[lab], 3) for lab in LABELS})
    row["span_micro_F1"] = round(s["micro"], 4)
    row.update({f"span_{lab}": round(s[lab], 3) for lab in ("PROBLEM", "ACTION")})
    return row


def train_crf(train, dev):
    from aerofix.segment.crf import CRFSegmenter
    best = None
    for c1 in (0.01, 0.1, 0.5):
        for c2 in (0.01, 0.1, 0.5):
            m = CRFSegmenter(c1, c2).fit(train)
            sc = token_macro([lab for _, lab in dev], m.predict([t for t, _ in dev]))
            if best is None or sc > best[0]:
                best = (sc, c1, c2, m)
    log.info("CRF best dev token macro-F1 %.4f at c1=%s c2=%s", *best[:3])
    return best[3], {"c1": best[1], "c2": best[2]}


def train_bert(train, dev, seed: int):
    from aerofix.segment.bert import BertSegmenter
    return BertSegmenter().fit(train, dev, seed=seed, score_fn=token_macro, log=log.info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+",
                    help="annotation files, later wins (default: segments_llm.jsonl then segments.jsonl)")
    ap.add_argument("--report", help="output markdown (default: reports/segmentation.md)")
    ap.add_argument("--models", help="model dir (default: models/segment)")
    ap.add_argument("--no-curve", action="store_true", help="skip the learning curve")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    from pathlib import Path
    adir = cfg["paths"]["annotations"]
    ann = [Path(p) for p in args.annotations] if args.annotations else [adir / "segments_llm.jsonl",
                                                                        adir / "segments.jsonl"]
    report = Path(args.report or cfg["paths"]["reports"] / "segmentation.md")
    mdir = Path(args.models or cfg["paths"]["models"] / "segment")

    data, sources = load_annotations(*ann)
    train, dev, test = data["train"], data["dev"], data["test"]
    log.info("annotated reports: train %d, dev %d, test %d", len(train), len(dev), len(test))
    gold = {"dev": [lab for _, lab in dev], "test": [lab for _, lab in test]}
    toks = {"dev": [t for t, _ in dev], "test": [t for t, _ in test]}

    rows, models, timing = [], {}, {}
    rows += [score_row("rules (baseline)", s, gold[s], rules.predict(toks[s])) for s in ("dev", "test")]

    t = time.perf_counter()
    crf, crf_params = train_crf(train, dev)
    timing["CRF"] = time.perf_counter() - t
    models["CRF"] = crf
    rows += [score_row(f"CRF (c1={crf_params['c1']}, c2={crf_params['c2']})", s, gold[s], crf.predict(toks[s]))
             for s in ("dev", "test")]

    t = time.perf_counter()
    bert = train_bert(train, dev, cfg["seed"])
    timing["DistilBERT"] = time.perf_counter() - t
    models["DistilBERT"] = bert
    rows += [score_row("DistilBERT (fine-tuned)", s, gold[s], bert.predict(toks[s])) for s in ("dev", "test")]

    table = pd.DataFrame(rows)
    dev_tab = table[table.split == "dev"].reset_index(drop=True)
    best_name = ["rules", "CRF", "DistilBERT"][int(dev_tab.span_micro_F1.idxmax())]
    log.info("best on dev span F1: %s", best_name)

    mdir.mkdir(parents=True, exist_ok=True)
    crf.save(mdir / "crf.pkl")
    bert.save(mdir / "bert")
    (mdir / "best.json").write_text(json.dumps({"best": best_name, "crf_params": crf_params}))

    # Learning curve: does more annotation help? (test scores, same dev for early stopping / tuning)
    curve = []
    if not args.no_curve:
        rng = random.Random(cfg["seed"])
        shuffled = rng.sample(train, len(train))
        for n in sorted({max(len(train) // 4, 1), len(train) // 2, len(train)}):
            sub = shuffled[:n]
            c, _ = train_crf(sub, dev)
            b = train_bert(sub, dev, cfg["seed"])
            for name, m in (("CRF", c), ("DistilBERT", b)):
                pred = m.predict(toks["test"])
                curve.append({"train_reports": n, "method": name,
                              "test_token_macro_F1": round(token_macro(gold["test"], pred), 4),
                              "test_span_micro_F1": round(span_scores(gold["test"], pred)["micro"], 4)})
            del b

    # Error analysis: test reports where the best model's spans differ from gold
    best_model = {"rules": None, "CRF": crf, "DistilBERT": bert}[best_name]
    pred_test = rules.predict(toks["test"]) if best_model is None else best_model.predict(toks["test"])
    errors = []
    for (tk, g), p in zip(test, pred_test):
        if spans(g) != spans(p):
            diff = [f"**{w}**[{a[0]}→{b[0]}]" if a != b else w for w, a, b in zip(tk, g, p)]
            errors.append(" ".join(diff))
    rng = random.Random(cfg["seed"])
    shown = rng.sample(errors, min(10, len(errors)))

    src = pd.Series(sources).rename("reports").rename_axis(["split", "annotator"]).reset_index()
    n_llm = int(src.reports[src.annotator != "human"].sum())
    rule_agree = [sum(a == b for a, b in zip(rules.predict([tk])[0], lab)) / max(len(lab), 1)
                  for tk, lab in train + dev + test]
    caveat = (f"**{n_llm} of {int(src.reports.sum())} reports were labelled by the local LLM** (silver labels, "
              "`python -m aerofix.segment.llm_label`), the rest by a human; human labels win where both exist. "
              "Scores therefore measure agreement with LLM labels where no human label exists: treat them as "
              "indicative until the test reports are human-verified.\n\n" if n_llm else "")

    md = ["# Problem / action segmentation (Phase 3)\n",
          "Generated by `python -m aerofix.segment.evaluate`. Model choices use dev only; the test set is "
          "scored once.\n",
          "## Annotation\n",
          caveat, md_table(src, index=False), "\n\n",
          f"Token-level agreement between the rule baseline and the gold labels: "
          f"{100 * sum(rule_agree) / max(len(rule_agree), 1):.1f}% on average.\n",
          "## Results\n",
          "Token F1 per label; span F1 counts a span as correct only if start, end and label all match.\n\n",
          md_table(table, index=False), "\n",
          f"\nBest on dev (span micro-F1): **{best_name}** → used to segment the corpus.\n",
          f"\nTraining time: CRF grid (9 fits) {timing['CRF']:.0f} s, DistilBERT {timing['DistilBERT']:.0f} s.\n"]
    if curve:
        md += ["\n## Learning curve (test)\n\n", md_table(pd.DataFrame(curve), index=False), "\n"]
    md += ["\n## Error examples (test, best model)\n\n",
           "Bold words are mislabelled; `[gold→predicted]` uses the first letter of each label "
           "(P=PROBLEM, A=ACTION, R=REF, O=OTHER).\n\n"]
    md += [f"{i + 1}. {e}\n" for i, e in enumerate(shown)]
    report.write_text("".join(md), encoding="utf-8")
    log.info("wrote %s", report)


if __name__ == "__main__":
    main()
