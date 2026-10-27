"""Evaluate Phase 2 normalisation: python -m aerofix.eval.normalisation

Writes reports/normalisation.md. Uses train for fitting and VAL for every number
(the 2025 test year is not touched).

1. Abbreviation coverage on val
2. LM perplexity on val: unigram/bigram/trigram x Laplace/Kneser-Ney
3. Spelling: synthetic typos injected into val (known ground truth),
   edit-distance-only vs noisy channel (edit + LM) over a grid of edit penalties;
   real corrections sample; out-of-dictionary rate before/after
4. Early ablation: TF-IDF + logistic regression, ATA top-K + OTHER, val macro-F1
"""
from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from aerofix.config import load_config
from aerofix.preprocess.abbrev import expand
from aerofix.preprocess.ngram_lm import NGramLM
from aerofix.preprocess.normalise import Normaliser
from aerofix.preprocess.tokenize import tokenize
from aerofix.utils import get_logger, md_table, set_seed

log = get_logger(__name__)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def corrupt(word: str, rng: random.Random) -> str:
    """One random edit: delete, insert, substitute or swap adjacent letters."""
    i = rng.randrange(len(word))
    op = rng.choice(["delete", "insert", "substitute", "transpose"])
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice(LETTERS) + word[i:]
    if op == "substitute":
        return word[:i] + rng.choice(LETTERS.replace(word[i], "")) + word[i + 1:]
    i = min(i, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def abbrev_section(norm: Normaliser, val_tok: list[list[str]]) -> str:
    applied, total = Counter(), 0
    for toks in val_tok:
        total += len(toks)
        applied.update(t for t in toks if t in norm.abbrev)
    n = sum(applied.values())
    top = pd.DataFrame([(a, " ".join(norm.abbrev[a]), c) for a, c in applied.most_common(15)],
                       columns=["abbrev", "expansion", "val_count"])
    return (f"## 1. Abbreviation expansion\n\n{len(norm.abbrev)} entries expand (ambiguous and keep-as-is rows "
            f"excluded). On val they rewrite **{n:,} of {total:,} tokens ({100 * n / total:.1f}%)**, "
            f"and {sum(1 for t in val_tok if any(x in norm.abbrev for x in t)) / len(val_tok):.0%} of reports "
            "contain at least one.\n\n" + md_table(top, index=False) + "\n")


def lm_section(train_docs, val_docs, cfg) -> str:
    rows = []
    for smoothing in ("laplace", "kn"):
        for order in (1, 2, 3):
            t = time.perf_counter()
            lm = NGramLM(order, smoothing, cfg["normalise"]["lm"]["discount"],
                         cfg["normalise"]["lm"]["unk_min_count"]).fit(train_docs)
            ppl = lm.perplexity(val_docs)
            rows.append({"model": f"{order}-gram", "smoothing": "Kneser-Ney" if smoothing == "kn" else "Laplace",
                         "val_perplexity": round(ppl, 1), "fit_s": round(time.perf_counter() - t)})
            log.info("LM %s", rows[-1])
            del lm
    return ("## 2. N-gram language model (val perplexity, lower is better)\n\n"
            f"Trained on {len(train_docs):,} train reports after abbreviation expansion; same vocabulary for all "
            "rows (train words seen once -> `<UNK>`), so the numbers are comparable.\n\n"
            + md_table(pd.DataFrame(rows), index=False) + "\n\n"
            "Laplace gets *worse* as the order grows: with a vocabulary of tens of thousands, adding 1 to "
            "every unseen trigram steals most of the probability mass. Kneser-Ney discounts seen counts "
            "a little and backs off to lower orders, so the trigram's extra context actually helps.\n")


PENALTY_GRID = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0)


def make_typo_cases(sc, val_tok: list[list[str]], seed: int, n_cases: int = 4000) -> list[tuple]:
    """(tokens, position, typo) triples: one real word per report corrupted by one edit."""
    rng = random.Random(seed)
    cases = []
    for toks in rng.sample(val_tok, k=min(len(val_tok), 4 * n_cases)):
        idx = [i for i, t in enumerate(toks) if t in sc.freq and len(t) >= 5 and t not in sc.protected]
        if not idx:
            continue
        i = rng.choice(idx)
        bad = corrupt(toks[i], rng)
        if bad in sc.freq or not bad.isalpha() or not sc.needs_correction(bad):
            continue  # typo is itself a valid word: undetectable by design, skip
        cases.append((toks, i, bad))
        if len(cases) == n_cases:
            break
    return cases


def typo_accuracy(sc, cases, mode: str, penalty: float | None = None) -> float:
    if penalty is not None:
        sc.penalty = penalty
    ok = 0
    for toks, i, bad in cases:
        right = toks[i + 1] if i + 1 < len(toks) else None
        ok += sc.best(bad, tuple(toks[max(0, i - 2):i]), right, mode) == toks[i]
    return ok / len(cases)


def spell_section(norm: Normaliser, val_tok: list[list[str]], cfg) -> tuple[str, float]:
    """Tune the edit penalty on one half of the synthetic typos, report on the other half,
    so the reported accuracy is not inflated by the tuning."""
    sc = norm.spell
    cases = make_typo_cases(sc, val_tok, cfg["seed"])
    tune, held = cases[: len(cases) // 2], cases[len(cases) // 2:]
    grid = [{"edit_penalty": p, "tune_accuracy_%": round(100 * typo_accuracy(sc, tune, "lm", p), 1)}
            for p in PENALTY_GRID]
    best_p = max(grid, key=lambda r: r["tune_accuracy_%"])["edit_penalty"]
    edit_acc = typo_accuracy(sc, held, "edit")
    lm_acc = typo_accuracy(sc, held, "lm", best_p)
    held_rows = pd.DataFrame([
        {"method": "edit distance only (ties -> most frequent)", "held_out_accuracy_%": round(100 * edit_acc, 1)},
        {"method": f"noisy channel: edit + trigram LM (penalty {best_p})", "held_out_accuracy_%": round(100 * lm_acc, 1)},
    ])
    diff = 100 * (lm_acc - edit_acc)
    verdict = (f"The language model adds {diff:+.1f} points: context helps choose between candidates at the "
               "same edit distance." if diff > 0.5 else
               f"The difference is {diff:+.1f} points, i.e. small: most single-edit typos have only one "
               "plausible dictionary neighbour, so context rarely changes the answer.")
    sc.penalty = cfg["normalise"]["spell"]["edit_penalty"]  # what the built corpus actually used

    # Real data: what does the corrector change on val, and how much OOV is left?
    changes, before, after, words = Counter(), 0, 0, 0
    for toks in val_tok[:20000]:
        fixed = sc.correct(toks)
        for a, b in zip(toks, fixed):
            if a.isalpha() and len(a) >= sc.min_len:
                words += 1
                before += a not in sc.freq
                after += b not in sc.freq
            if a != b:
                changes[(a, b)] += 1
    sample = pd.DataFrame([(a, b, c) for (a, b), c in changes.most_common(50)],
                          columns=["original", "corrected", "count_in_20k_val_reports"])
    text = (f"## 3. Spelling correction\n\n**Synthetic test:** {len(cases):,} val reports, one real word per report "
            "corrupted by one random edit (delete / insert / substitute / swap). The corrector sees the "
            "typo in its real context and must recover the original word. The edit penalty is tuned on "
            f"the first {len(tune):,} cases and accuracy is reported on the other {len(held):,}.\n\n"
            "Tuning grid (tune half):\n\n" + md_table(pd.DataFrame(grid), index=False) +
            f"\n\nHeld-out half:\n\n" + md_table(held_rows, index=False) + f"\n\n{verdict} "
            f"The config uses edit_penalty = {cfg['normalise']['spell']['edit_penalty']}.\n\n"
            "Caveat: synthetic single-edit typos are easier than real ones (real typos are often 2+ edits or "
            "split/merged words), so these accuracies are an upper bound.\n\n"
            f"**Real val text** (first 20,000 val reports): words of length >= {sc.min_len} not in the train "
            f"dictionary: **{100 * before / words:.2f}% before -> {100 * after / words:.2f}% after** correction. "
            f"{sum(changes.values()):,} tokens changed. The 50 most frequent changes, for manual spot-checking:\n\n"
            + md_table(sample, index=False) + "\n")
    return text, best_p


def ablation_section(df: pd.DataFrame, norm: Normaliser, cfg) -> str:
    rng = np.random.default_rng(cfg["seed"])
    train = df[df.split == "train"]
    train = train.iloc[rng.choice(len(train), size=min(cfg["normalise"]["ablation_train_sample"], len(train)),
                                  replace=False)]
    val = df[df.split == "val"]
    top = df[df.split == "train"].ata2.value_counts().index[: cfg["classify"]["top_k_ata2"]]
    label = lambda s: np.where(s.isin(top), s, "OTHER")
    variants = {
        "masked raw (tokenised only)": lambda d: [" ".join(tokenize(t)) for t in d.text_masked],
        "+ abbreviations": lambda d: [" ".join(expand(tokenize(t), norm.abbrev)) for t in d.text_masked],
        "+ abbreviations + spelling": lambda d: list(d.text_norm),
        "+ abbrev + spelling + Porter stem": lambda d: list(d.text_stem),
        "+ abbrev + spelling + WordNet lemma": lambda d: list(d.text_lemma),
    }
    rows = []
    for name, fn in variants.items():
        t = time.perf_counter()
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, max_features=300_000)
        xtr, xva = vec.fit_transform(fn(train)), vec.transform(fn(val))
        unigrams = sum(1 for f in vec.vocabulary_ if " " not in f)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
        clf.fit(xtr, label(train.ata2))
        pred = clf.predict(xva)
        rows.append({"input": name, "unigram_vocab": unigrams,
                     "val_macro_F1": round(f1_score(label(val.ata2), pred, average="macro"), 4),
                     "val_accuracy": round(accuracy_score(label(val.ata2), pred), 4),
                     "fit_s": round(time.perf_counter() - t)})
        log.info("ablation %s", rows[-1])
    f1 = [r["val_macro_F1"] for r in rows]
    spread = max(f1) - min(f1)
    finding = (f"**Finding:** all variants are within {spread:.3f} macro-F1 of each other. For a linear TF-IDF "
               "model trained on 100k reports, normalisation does not measurably help classification: each "
               "spelling variant is frequent enough to get its own weight, and stemming/lemmatisation only "
               "shrinks the vocabulary. Normalisation is expected to matter more where vocabulary mismatch "
               "hurts directly: BM25 retrieval of short technician queries (measured in Phase 5) and the "
               "Word2Vec/BiLSTM model (Phase 4)." if spread < 0.005 else
               f"**Finding:** variants differ by up to {spread:.3f} macro-F1; see the table for which step helps.")
    return ("## 4. Early ablation (preliminary; final numbers in Phase 4)\n\n"
            f"TF-IDF (word 1-2 grams) + logistic regression (balanced class weights), ATA top-"
            f"{cfg['classify']['top_k_ata2']} + OTHER. Trained on a random {len(train):,}-report train sample, "
            "scored on the full 2024 val year. Input is the **whole masked report** (problem + action); "
            "Phase 4 uses the problem part only, so absolute numbers will change.\n\n"
            + md_table(pd.DataFrame(rows), index=False) + "\n\n" + finding + "\n")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed"]
    df = pd.read_parquet(proc / "sdr_a320_b737.parquet", columns=["row_id", "text_masked", "split", "ata2"])
    df = df.merge(pd.read_parquet(proc / "sdr_norm.parquet"), on="row_id")
    norm = Normaliser.load(cfg["paths"]["models"] / "normaliser.pkl")

    train_tok = [expand(tokenize(t), norm.abbrev) for t in df.text_masked[df.split == "train"]]
    val_tok = [expand(tokenize(t), norm.abbrev) for t in df.text_masked[df.split == "val"]]

    parts = ["# Text normalisation (Phase 2)\n",
             "Generated by `python -m aerofix.eval.normalisation`. Everything is fitted on train (<= 2023) "
             "and measured on val (2024); the 2025 test year is not used.\n"]
    parts.append(abbrev_section(norm, [tokenize(t) for t in df.text_masked[df.split == "val"]]))
    spell_text, best_penalty = spell_section(norm, val_tok, cfg)
    parts.append(lm_section(train_tok, val_tok, cfg))
    parts.append(spell_text)
    del train_tok
    parts.append(ablation_section(df, norm, cfg))

    out = cfg["paths"]["reports"] / "normalisation.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote %s (best edit penalty %.1f)", out, best_penalty)


if __name__ == "__main__":
    main()
