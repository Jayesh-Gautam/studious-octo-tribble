"""TF-IDF + logistic regression: python -m aerofix.classify.tfidf [--skip-ablations]

Features: word 1-2 grams (our domain tokeniser, so R&R and P/N stay whole) plus
character 3-5 grams inside word boundaries. Character n-grams make the model
robust to typos and abbreviations: CORRISION and CORROSION share most of their
3-5 grams even though they are different words.

Main models (one per target) are trained on the full train split; C is chosen on
val ata2 macro-F1 and reused for the other targets. Ablations (ata2 only) use a
fixed random train sample for runtime, and report on val:
  leakage:        full report unmasked / masked, problem unmasked / masked
  normalisation:  problem masked raw / + abbrev + spelling / + stem / + lemma
  recent years:   2015-2023 sample vs all-years sample of the same size
  cross-family:   train A320 -> eval B737 and vice versa, vs in-family (equal train size)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion

from aerofix.classify.data import TARGETS, LabelSpace, load_frame, save_preds
from aerofix.classify.metrics import summary
from aerofix.config import load_config
from aerofix.preprocess.tokenize import tokenize
from aerofix.utils import get_logger, set_seed

log = get_logger(__name__)


def tokens_joined(text: str) -> str:
    """Char n-grams are taken from the tokenised text (punctuation dropped). On val this beat raw text by
    ~2 macro-F1 points: raw 'LEAKING.' yields n-grams containing '.', splitting evidence across variants."""
    return " ".join(tokenize(text))


def make_vectorizer(cfg: dict) -> FeatureUnion:
    t = cfg["classify"]["tfidf"]
    word = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                           min_df=t["min_df"], max_features=t["word_max_features"], sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", preprocessor=tokens_joined, ngram_range=(3, 5), lowercase=False,
                           min_df=t["min_df"], max_features=t["char_max_features"], sublinear_tf=True,
                           dtype=np.float32)
    return FeatureUnion([("word", word), ("char", char)])


def fit_lr(x, y, C: float, seed: int = 42):
    """Multinomial LR (L-BFGS) for few classes. For many classes (ata4: 226) L-BFGS needs
    ~10 GB of history for 55M weights, so we switch to SGD logistic regression
    (one-vs-rest, parallel over classes) with the same regularisation strength:
    alpha = 1 / (C * n_samples) is the SGD equivalent of LR's C."""
    if len(np.unique(y)) > 50:
        return SGDClassifier(loss="log_loss", alpha=1.0 / (C * x.shape[0]), class_weight="balanced", max_iter=30,
                             tol=1e-4, n_jobs=-1, random_state=seed).fit(x, y)
    return LogisticRegression(C=C, class_weight="balanced", max_iter=300, tol=1e-3).fit(x, y)


def full_probs(clf: LogisticRegression, x, n_classes: int) -> np.ndarray:
    """predict_proba only has columns for classes seen in training; pad the rest with 0."""
    p = np.zeros((x.shape[0], n_classes), dtype=np.float32)
    p[:, clf.classes_] = clf.predict_proba(x)
    return p


def run_variant(name: str, train: pd.DataFrame, evals: dict[str, pd.DataFrame], col: str, labels: LabelSpace,
                C: float, cfg: dict) -> list[dict]:
    t = time.perf_counter()
    vec = make_vectorizer(cfg)
    xtr = vec.fit_transform(train[col].fillna(""))
    clf = fit_lr(xtr, labels.encode("ata2", train.ata2), C)
    rows = []
    for ename, ev in evals.items():
        p = full_probs(clf, vec.transform(ev[col].fillna("")), len(labels.names("ata2")))
        rows.append({"experiment": name, "eval_on": ename, "train_rows": len(train),
                     **summary(labels.encode("ata2", ev.ata2), p), "fit_s": round(time.perf_counter() - t)})
        log.info("%s", rows[-1])
    return rows


def ablations(df: pd.DataFrame, labels: LabelSpace, C: float, cfg: dict) -> pd.DataFrame:
    n = cfg["classify"]["tfidf"]["ablation_train_sample"]
    seed = cfg["seed"]
    train, val = df[df.split == "train"], df[df.split == "val"]
    sample = train.sample(min(n, len(train)), random_state=seed)
    rows = []
    for col, name in [("text", "leakage: full report, UNMASKED"), ("text_masked", "leakage: full report, masked"),
                      ("problem_text", "leakage: problem only, unmasked"), ("problem_masked", "leakage: problem only, masked")]:
        rows += run_variant(name, sample, {"val": val}, col, labels, C, cfg)
    # Control: tokenised only (punctuation dropped, same tokeniser as the normalised columns), so the
    # normalisation rows below differ from it ONLY by abbreviations / spelling / stemming / lemmatisation.
    sample, val = sample.copy(), val.copy()
    for d in (sample, val):
        d["problem_tok"] = [" ".join(tokenize(t)) for t in d.problem_masked.fillna("")]
    rows += run_variant("norm: tokenised only (control)", sample, {"val": val}, "problem_tok", labels, C, cfg)
    for col, name in [("problem_norm", "norm: + abbrev + spelling"), ("problem_stem", "norm: + Porter stem"),
                      ("problem_lemma", "norm: + WordNet lemma")]:
        rows += run_variant(name, sample, {"val": val}, col, labels, C, cfg)
    recent = train[train.file_year >= 2015]
    rows += run_variant("years: 2015-2023 only", recent.sample(min(n, len(recent)), random_state=seed),
                        {"val": val}, "problem_masked", labels, C, cfg)
    # Cross-family: equal train size (the smaller family's train size, capped at the sample size).
    fam_n = min(n, (train.family == "A320").sum(), (train.family == "B737").sum())
    vals = {f"val {f}": val[val.family == f] for f in ("A320", "B737")}
    for fam in ("A320", "B737"):
        tr = train[train.family == fam].sample(fam_n, random_state=seed)
        rows += run_variant(f"family: train {fam} only", tr, vals, "problem_masked", labels, C, cfg)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--ablations-only", action="store_true", help="reuse saved labels + best C, rerun ablations")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    out = cfg["paths"]["models"] / "classify"
    df = load_frame(cfg)
    if args.ablations_only:
        best_C = json.loads((out / "tfidf_lr" / "tuning.json").read_text())["best_C"]
        abl = ablations(df, LabelSpace.load(out / "labels.json"), best_C, cfg)
        abl.to_csv(out / "ablations_tfidf.csv", index=False)
        return
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    labels = LabelSpace.fit(train, cfg)
    labels.save(out / "labels.json")
    col = cfg["classify"]["text_col"]
    log.info("classes: %s", {t: len(labels.names(t)) for t in TARGETS})

    t0 = time.perf_counter()
    vec = make_vectorizer(cfg)
    xtr = vec.fit_transform(train[col])
    xva, xte = vec.transform(val[col]), vec.transform(test[col])
    log.info("vectorised: %s features in %.0f s", xtr.shape[1], time.perf_counter() - t0)

    ytr = labels.encode("ata2", train.ata2)
    yva = labels.encode("ata2", val.ata2)
    tuning = []
    for C in cfg["classify"]["tfidf"]["C_grid"]:
        t = time.perf_counter()
        clf = fit_lr(xtr, ytr, C)
        s = summary(yva, full_probs(clf, xva, len(labels.names("ata2"))))
        tuning.append({"C": C, **s, "fit_s": round(time.perf_counter() - t)})
        log.info("C=%s %s", C, tuning[-1])
    best_C = max(tuning, key=lambda r: r["macro_F1"])["C"]

    probs = {"val": {}, "test": {}}
    models = {}
    for target in TARGETS:
        t = time.perf_counter()
        clf = fit_lr(xtr, labels.encode(target, train[target]), best_C)
        models[target] = clf
        for split, x in (("val", xva), ("test", xte)):
            probs[split][target] = full_probs(clf, x, len(labels.names(target)))
        log.info("%s fitted in %.0f s, val %s", target, time.perf_counter() - t,
                 summary(labels.encode(target, val[target]), probs["val"][target]))
    mdir = out / "tfidf_lr"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "models": models}, mdir / "model.joblib", compress=3)
    save_preds(mdir, "val", val.row_id.values, probs["val"])
    save_preds(mdir, "test", test.row_id.values, probs["test"])
    (mdir / "tuning.json").write_text(json.dumps({"C_grid": tuning, "best_C": best_C}))

    if not args.skip_ablations:
        abl = ablations(df, labels, best_C, cfg)
        abl.to_csv(out / "ablations_tfidf.csv", index=False)
    log.info("done in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
