"""Simulated feedback / learning loop: python -m aerofix.feedback.simulate

Deployment story: models are trained on reports up to 2023 and go live in 2024.
Each quarter, technicians' corrections come back as feedback:
    round 1 = Jan-Mar 2024, round 2 = Apr-Jun, round 3 = Jul-Sep
(Oct-Dec 2024 is held out for training-time monitoring). After each round:
  1. the round's feedback is written to the SQLite store (source='simulated'),
  2. it is read back from the store and used to update the models:
     - classifier: continue fine-tuning the previous round's DistilBERT for one epoch
       on the new feedback + a 50k random "replay" sample of the original training
       data (replay prevents catastrophic forgetting of older fault types)
     - CONTROL: the same amount of training on replay data only, so any gain of the
       feedback run over the control comes from the feedback, not from extra steps
     - retrieval: the feedback reports join the search index ("index growth"), and
       the bi-encoder is fine-tuned on triples mined from the feedback ("+ fine-tune")
  3. everything is measured on 2,000 fixed 2025 test reports (reporting only; no
     decision in this script uses the test year).
Production models are not modified; round models go to models/feedback_loop/.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from aerofix.classify import neural
from aerofix.classify.data import TARGETS, LabelSpace, class_weights
from aerofix.classify.metrics import summary
from aerofix.classify.transformer import MultiTaskTransformer
from aerofix.config import load_config
from aerofix.feedback import store
from aerofix.retrieve.common import gains, load_corpus, metrics_for
from aerofix.retrieve.index import DenseIndex
from aerofix.retrieve.train import encode, train_biencoder
from aerofix.utils import get_logger, md_table, set_seed

log = get_logger(__name__)
ROUNDS = {1: (1, 3), 2: (4, 6), 3: (7, 9)}  # months of 2024
REPLAY = 50_000
COLORS = {"feedback": "#2a78d6", "control": "#eb6834"}  # validated palette slots 1-2 (see Phase 1 EDA)


def classifier_round(model, tok, train_rows, monitor, labels, cfg, seed):
    tc = cfg["classify"]["transformer"]
    enc = lambda texts: tok([t.lower() for t in texts], truncation=True, max_length=tc["max_seq_len"])["input_ids"]
    ys = {t: train_rows[f"y_{t}"].to_numpy(dtype=np.int64) for t in TARGETS}
    weights = {t: class_weights(ys[t][ys[t] >= 0], len(labels.names(t)), cfg["classify"]["class_weight_power"])
               for t in TARGETS}
    mys = {t: labels.encode(t, monitor[t]) for t in TARGETS}

    def make_opt(m, total, warmup):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.01)  # lower LR: updating, not re-learning
        return opt, get_linear_schedule_with_warmup(opt, warmup, total)

    model, hist = neural.train(model, enc(list(train_rows.text)), ys, enc(list(monitor.problem_masked)), mys, weights,
                               pad_id=tok.pad_token_id, device="cuda" if torch.cuda.is_available() else "cpu",
                               epochs=1, batch_size=tc["batch_size"], make_optimizer=make_opt, warmup_frac=0.06,
                               patience=1, seed=seed, log=log.info)
    return model


def eval_classifier(model, tok, test, labels, cfg):
    seqs = tok([t.lower() for t in test.problem_masked], truncation=True,
               max_length=cfg["classify"]["transformer"]["max_seq_len"])["input_ids"]
    probs = neural.predict(model, seqs, tok.pad_token_id, "cuda" if torch.cuda.is_available() else "cpu")
    return {t: summary(labels.encode(t, test[t]), probs[t]) for t in TARGETS}


def feedback_triples(pool: pd.DataFrame, index: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Each feedback report is an anchor; positive = an index report with the same family + JASC + condition,
    hard negative = same family + chapter, different JASC (same scheme as Phase 5)."""
    rng = np.random.default_rng(seed)
    key = lambda d: d.family + "|" + d.ata4 + "|" + d.condition
    groups = index.groupby(key(index)).indices
    chapters = index.groupby(index.family + "|" + index.ata2).indices
    texts, ata4 = index.problem_masked.values, index.ata4.values
    rows = []
    for r, k in zip(pool.itertuples(), key(pool)):
        pos = groups.get(k)
        neg_pool = chapters.get(f"{r.family}|{r.ata2}")
        if pos is None or neg_pool is None or not r.problem_masked:
            continue
        p = rng.choice(pos)
        cands = neg_pool[ata4[neg_pool] != r.ata4]
        if texts[p] == r.problem_masked or not len(cands):
            continue
        rows.append((k, r.problem_masked, texts[p], texts[rng.choice(cands)]))
    return pd.DataFrame(rows, columns=["group", "anchor", "positive", "negative"])


def eval_retrieval(vecs, index_rows, queries, qvecs, corpus):
    idx = DenseIndex(vecs, corpus.family.values[index_rows])
    found = idx.search(qvecs, queries.family.values, 10)
    docs = corpus.iloc[index_rows].reset_index(drop=True)
    m = [metrics_for(gains(q, docs.iloc[r])) for (_, q), r in zip(queries.iterrows(), found)]
    return {k: round(float(np.mean([x[k] for x in m])), 4) for k in ("Hit@5", "P@5", "NDCG@5")}


def write_report(res: pd.DataFrame, cfg: dict, out) -> None:
    """Three panels, each with feedback vs control, validated palette, one y-axis per panel. Every panel spans
    at least 0.03 so sub-point noise is not drawn as a dramatic swing."""
    panels = [("clf_feedback_macroF1", "clf_control_macroF1", "ATA chapter: test macro-F1",
               "feedback + replay", "control: replay only"),
              ("cond_feedback_macroF1", "cond_control_macroF1", "Part condition: test macro-F1",
               "feedback + replay", "control: replay only"),
              ("ret_growth_finetune_Hit@5", "ret_growth_Hit@5", "Similar-case retrieval: test Hit@5",
               "index growth + bi-encoder fine-tune", "index growth only")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (fb_col, ctl_col, ttl, fb_lab, ctl_lab) in zip(axes, panels):
        ax.plot(res["round"], res[fb_col], color=COLORS["feedback"], marker="o", markersize=5, linewidth=2, label=fb_lab)
        ax.plot(res["round"], res[ctl_col], color=COLORS["control"], marker="o", markersize=5, linewidth=2, label=ctl_lab)
        lo, hi = min(res[fb_col].min(), res[ctl_col].min()), max(res[fb_col].max(), res[ctl_col].max())
        if hi - lo < 0.03:
            mid = (hi + lo) / 2
            lo, hi = mid - 0.015, mid + 0.015
        ax.set_ylim(lo - 0.002, hi + 0.002)
        ax.set_title(ttl, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xticks(res["round"], [f"R{r}" for r in res["round"]])
        ax.set_xlabel("feedback round (R0 = deployed model)", color="#52514e")
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(cfg["paths"]["reports"] / "learning_curve.png", dpi=150)
    plt.close(fig)

    show = res[["round", "feedback_rows", "index_size", "clf_feedback_macroF1", "clf_control_macroF1",
                "clf_feedback_acc", "cond_feedback_macroF1", "cond_control_macroF1", "ret_growth_Hit@5",
                "ret_growth_finetune_Hit@5", "ret_growth_NDCG@5", "ret_growth_finetune_NDCG@5"]]
    last, first = res.iloc[-1], res.iloc[0]
    d = {"ata_feedback_minus_control": last.clf_feedback_macroF1 - last.clf_control_macroF1,
         "condition_feedback_minus_control": last.cond_feedback_macroF1 - last.cond_control_macroF1,
         "condition_feedback_vs_round0": last.cond_feedback_macroF1 - first.cond_feedback_macroF1,
         "retrieval_hit5_index_growth": last["ret_growth_Hit@5"] - first["ret_growth_Hit@5"],
         "retrieval_hit5_finetune_extra": last["ret_growth_finetune_Hit@5"] - last["ret_growth_Hit@5"]}
    md = ["# Feedback learning loop (Phase 9)\n\n",
          "Generated by `python -m aerofix.feedback.simulate`. Simulated deployment: models trained on <= 2023; "
          "technician feedback arrives quarterly in 2024 (R1 Jan-Mar, R2 Apr-Jun, R3 Jul-Sep) and is written to and "
          "read back from the SQLite feedback store. Measured on 2,000 fixed 2025 test reports (reporting only). "
          "The control receives the same number of training steps on old (<= 2023) data, so the gap between the "
          "two classifier lines is the value of the 2024 feedback itself.\n\n",
          "![learning curve](learning_curve.png)\n\n", md_table(show, index=False), "\n\n",
          "## After 3 rounds\n\n",
          f"- **Part condition:** feedback {d['condition_feedback_minus_control']:+.4f} macro-F1 vs the control "
          f"({d['condition_feedback_vs_round0']:+.4f} vs the deployed model). Condition codes drift between years "
          "(Phase 4: OTHER grows from 15% of train to 24% of test), and only the feedback carries the newer usage.\n",
          f"- **ATA chapter:** feedback {d['ata_feedback_minus_control']:+.4f} vs the control: no measurable effect. "
          "The chapter classifier is already at ~0.86 and chapter labels do not drift, so extra recent examples "
          "add little; both updated models move by < 1 point (noise level for 2,000 queries).\n",
          f"- **Retrieval Hit@5:** {d['retrieval_hit5_index_growth']:+.4f} from adding the feedback reports to the "
          f"index alone; fine-tuning the bi-encoder on feedback triples adds {d['retrieval_hit5_finetune_extra']:+.4f}. "
          "Most of the benefit of feedback for retrieval is simply having recent, similar cases to find.\n"]
    (cfg["paths"]["reports"] / "feedback_loop.md").write_text("".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(d))
    log.info("wrote reports/feedback_loop.md and reports/learning_curve.png\n%s", show.to_string())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="redraw plot + report from learning_curve.csv")
    args = ap.parse_args()
    cfg = load_config()
    if args.report_only:
        out = cfg["paths"]["models"] / "feedback_loop"
        write_report(pd.read_csv(out / "learning_curve.csv"), cfg, out)
        return
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = cfg["paths"]["models"] / "feedback_loop"
    out.mkdir(parents=True, exist_ok=True)
    labels = LabelSpace.load(cfg["paths"]["models"] / "classify" / "labels.json")

    corpus = load_corpus(cfg)
    dates = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet", columns=["row_id", "difficulty_date"])
    corpus = corpus.merge(dates, on="row_id")
    corpus["month"] = corpus.difficulty_date.dt.month
    assert (corpus.row_id.values == np.arange(len(corpus))).all()
    for t in TARGETS:
        corpus[f"y_{t}"] = labels.encode(t, corpus[t])
    train = corpus[corpus.split == "train"]
    val = corpus[corpus.split == "val"]
    monitor = val[val.month >= 10].sample(5000, random_state=cfg["seed"])  # Oct-Dec 2024: training-time check only
    test = corpus[corpus.split == "test"].sample(2000, random_state=cfg["seed"])

    # ---- round 0: production models ----
    cdir = cfg["paths"]["models"] / "classify" / "distilbert"
    tok = AutoTokenizer.from_pretrained(str(cdir / "tokenizer"))
    n_classes = {t: len(labels.names(t)) for t in TARGETS}

    def load_clf():
        m = MultiTaskTransformer("distilbert-base-uncased", n_classes)
        m.load_state_dict(torch.load(cdir / "model.pt", map_location="cpu"))
        return m.to("cuda" if torch.cuda.is_available() else "cpu")

    clf_fb, clf_ctrl = load_clf(), load_clf()
    vec0 = np.load(cfg["paths"]["models"] / "retrieve" / "vectors_ft.npy").astype(np.float32)
    bienc = SentenceTransformer(str(cfg["paths"]["models"] / "retrieve" / "biencoder"))
    index_rows = np.where(corpus.split == "train")[0]

    store.delete_source("simulated")  # a re-run starts from a clean simulated history
    r0 = eval_classifier(clf_fb, tok, test, labels, cfg)
    ret0 = eval_retrieval(vec0[index_rows], index_rows, test, vec0[test.row_id.values], corpus)
    rows = [{"round": 0, "feedback_rows": 0, "index_size": len(index_rows),
             "clf_feedback_macroF1": r0["ata2"]["macro_F1"], "clf_feedback_acc": r0["ata2"]["accuracy"],
             "clf_control_macroF1": r0["ata2"]["macro_F1"], "cond_feedback_macroF1": r0["condition"]["macro_F1"],
             "cond_control_macroF1": r0["condition"]["macro_F1"],
             "ret_growth_Hit@5": ret0["Hit@5"], "ret_growth_finetune_Hit@5": ret0["Hit@5"],
             "ret_growth_NDCG@5": ret0["NDCG@5"], "ret_growth_finetune_NDCG@5": ret0["NDCG@5"]}]
    log.info("round 0: %s", rows[-1])

    total_fb = 0
    for rnd, (m0, m1) in ROUNDS.items():
        t0 = time.perf_counter()
        pool = val[(val.month >= m0) & (val.month <= m1)]
        # 1. technicians' corrections -> SQLite (the true labels of the filed report)
        store.add_many([{"source": "simulated", "round": rnd, "family": r.family, "text": r.problem_masked,
                         "correct_ata2": r.ata2, "system_correct": None, "sdr_row_id": int(r.row_id)}
                        for r in pool.itertuples()])
        # 2. read this round's feedback back from the store
        fb = store.all_rows()
        fb = fb[(fb.source == "simulated") & (fb["round"] == rnd)]
        lab = corpus.loc[fb.sdr_row_id.astype(int).values]
        fb_rows = pd.DataFrame({"text": fb.text.values, "y_ata2": labels.encode("ata2", fb.correct_ata2),
                                "y_ata4": lab.y_ata4.values, "y_condition": lab.y_condition.values})
        total_fb += len(fb_rows)
        replay = train.sample(REPLAY, random_state=cfg["seed"] + rnd)
        as_rows = lambda d: pd.DataFrame({"text": d.problem_masked.values, **{f"y_{t}": d[f"y_{t}"].values for t in TARGETS}})
        clf_fb = classifier_round(clf_fb, tok, pd.concat([fb_rows, as_rows(replay)]), monitor, labels, cfg, cfg["seed"] + rnd)
        extra = train.drop(replay.index).sample(len(fb_rows), random_state=cfg["seed"] + 100 + rnd)
        clf_ctrl = classifier_round(clf_ctrl, tok, as_rows(pd.concat([replay, extra])), monitor, labels, cfg, cfg["seed"] + rnd)
        rf, rc = eval_classifier(clf_fb, tok, test, labels, cfg), eval_classifier(clf_ctrl, tok, test, labels, cfg)

        # 3. retrieval: index growth (feedback reports become searchable) and bi-encoder fine-tuning
        index_rows = np.concatenate([index_rows, pool.row_id.values])
        growth = eval_retrieval(vec0[index_rows], index_rows, test, vec0[test.row_id.values], corpus)
        triples = feedback_triples(pool, corpus.iloc[index_rows].reset_index(drop=True), cfg["seed"] + rnd)
        bienc = train_biencoder(bienc, triples, lr=1e-5, seed=cfg["seed"] + rnd) if len(triples) >= 64 else bienc
        vec_ft = encode(bienc, list(corpus.problem_masked.values[index_rows]))
        q_ft = encode(bienc, list(test.problem_masked))
        tuned = eval_retrieval(vec_ft, index_rows, test, q_ft, corpus)

        rows.append({"round": rnd, "feedback_rows": total_fb, "index_size": len(index_rows),
                     "clf_feedback_macroF1": rf["ata2"]["macro_F1"], "clf_feedback_acc": rf["ata2"]["accuracy"],
                     "clf_control_macroF1": rc["ata2"]["macro_F1"], "cond_feedback_macroF1": rf["condition"]["macro_F1"],
                     "cond_control_macroF1": rc["condition"]["macro_F1"],
                     "ret_growth_Hit@5": growth["Hit@5"], "ret_growth_finetune_Hit@5": tuned["Hit@5"],
                     "ret_growth_NDCG@5": growth["NDCG@5"], "ret_growth_finetune_NDCG@5": tuned["NDCG@5"],
                     "anchor_triples": len(triples), "minutes": round((time.perf_counter() - t0) / 60, 1)})
        log.info("round %d: %s", rnd, rows[-1])
        torch.save(clf_fb.state_dict(), out / f"classifier_round{rnd}.pt")

    res = pd.DataFrame(rows)
    res.to_csv(out / "learning_curve.csv", index=False)

    write_report(res, cfg, out)


if __name__ == "__main__":
    main()
