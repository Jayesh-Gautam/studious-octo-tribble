"""Build canonical actions and evaluate both recommenders: python -m aerofix.recommend.evaluate

1. Canonical actions for every report (rules + clustering, fitted on train) ->
   data/processed/sdr_actions.parquet, models/recommend/object_map.json
2. 2,000 test (2025) queries; the Phase 5 retriever searches train+val, top-k cases.
3. Actions: hit@1 / hit@3 = one of the top-k recommended canonical actions appears in
   the report's own action text (also at verb level). Baselines: globally most
   frequent actions, and most frequent actions for the JASC code predicted by the
   Phase 4 DistilBERT classifier.
4. References: hit@1 / hit@3 = a recommended reference is one the report actually
   cited (exact "AMM 52-10-00"). Methods: global most cited, retrieval + count,
   retrieval + specificity, direct TF-IDF classifier. Also the share of top-1
   recommendations that are the single most generic reference (SRM 51-40-02).
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from aerofix.classify.data import LabelSpace
from aerofix.classify.tfidf import make_vectorizer
from aerofix.config import load_config
from aerofix.preprocess.normalise import Normaliser
from aerofix.recommend import references as R
from aerofix.recommend.actions import action_tokens, aggregate, canonicalise, cluster_objects, extract_actions, split_action
from aerofix.retrieve.common import load_corpus
from aerofix.retrieve.search import Retriever
from aerofix.utils import get_logger, md_table, set_seed

log = get_logger(__name__)
K = 20


def hits(recs: list[list[str]], truth: list[set[str]]) -> dict:
    h1 = np.mean([bool(r[:1]) and r[0] in t for r, t in zip(recs, truth)])
    h3 = np.mean([bool(set(r[:3]) & t) for r, t in zip(recs, truth)])
    return {"hit@1": round(float(h1), 4), "hit@3": round(float(h3), 4)}


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = cfg["paths"]["models"] / "recommend"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_corpus(cfg)
    main_tab = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet", columns=["row_id", "refs"])
    df = df.merge(main_tab, on="row_id")
    norm = Normaliser.load(cfg["paths"]["models"] / "normaliser.pkl")

    # ---- 1. canonical actions ----
    t = time.perf_counter()
    raw = [extract_actions(action_tokens(a, norm.abbrev)) for a in df.action_text.fillna("")]
    train_mask = (df.split == "train").values
    objects = Counter(split_action(a)[1] for acts, m in zip(raw, train_mask) if m for a in acts)
    from sentence_transformers import SentenceTransformer
    from aerofix.retrieve.train import encode
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    mapping = cluster_objects(objects, lambda xs: encode(st, xs))
    df["actions"] = [canonicalise(a, mapping) for a in raw]
    (out_dir / "object_map.json").write_text(json.dumps(mapping, indent=0))
    df[["row_id", "actions"]].to_parquet(cfg["paths"]["processed"] / "sdr_actions.parquet", index=False)
    n_with = float((df.actions.str.len() > 0).mean())
    log.info("actions extracted in %.0f s; %d object merges; %.1f%% of reports have >= 1 action",
             time.perf_counter() - t, len(mapping), 100 * n_with)
    merges = pd.Series(mapping).rename("merged into").rename_axis("object").reset_index()
    merges["count"] = merges.object.map(objects)
    merges = merges.sort_values("count", ascending=False).head(15)

    # ---- 2. retrieval for test queries ----
    vec = np.load(cfg["paths"]["models"] / "retrieve" / "vectors_ft.npy").astype(np.float32)
    doc_idx = np.where(df.split.isin(["train", "val"]))[0]
    docs = df.iloc[doc_idx].reset_index(drop=True)
    retr = Retriever(docs, vec[doc_idx], cfg)
    rng = np.random.default_rng(cfg["seed"])
    q_idx = rng.choice(np.where(df.split == "test")[0], 2000, replace=False)
    queries = df.iloc[q_idx].reset_index(drop=True)
    t = time.perf_counter()
    found = [retr.search(q.problem_masked, q.problem_lemma, q.family, k=K, qvec=vec[i])
             for i, q in zip(q_idx, queries.itertuples())]
    ms = 1000 * (time.perf_counter() - t) / len(queries)
    log.info("retrieval: %.1f ms per query", ms)

    train = df[train_mask]
    # ---- 3. actions ----
    has_act = queries.actions.str.len() > 0
    truth = [set(a) for a in queries.actions]
    truth_verb = [{split_action(x)[0] for x in a} for a in queries.actions]
    glob = [a for a, _ in Counter(a for acts in train.actions for a in acts).most_common(5)]
    by_ata4 = {k: [a for a, _ in Counter(x for acts in g for x in acts).most_common(5)]
               for k, g in train.groupby(train.family + "|" + train.ata4).actions}
    labels = LabelSpace.load(cfg["paths"]["models"] / "classify" / "labels.json")
    pz = np.load(cfg["paths"]["models"] / "classify" / "distilbert" / "preds_test.npz")
    pred4 = dict(zip(pz["row_id"], np.array(labels.names("ata4"))[pz["ata4"].astype(np.float32).argmax(1)]))
    act_methods = {
        "Global most frequent actions": [glob] * len(queries),
        "Most frequent for predicted JASC (DistilBERT)": [by_ata4.get(f"{q.family}|{pred4.get(q.row_id)}", glob)
                                                          for q in queries.itertuples()],
        f"Retrieval vote, top {K} (similarity-weighted)": [[a for a, _ in aggregate(list(docs.actions.values[ix]), s)]
                                                           for ix, s in found],
        "Retrieval vote, top 5": [[a for a, _ in aggregate(list(docs.actions.values[ix[:5]]), s[:5])] for ix, s in found],
    }
    act_rows = []
    for name, recs in act_methods.items():
        r = [recs[i] for i in np.where(has_act)[0]]
        t_ = [truth[i] for i in np.where(has_act)[0]]
        tv = [truth_verb[i] for i in np.where(has_act)[0]]
        verb_recs = [list(dict.fromkeys(split_action(a)[0] for a in x)) for x in r]
        act_rows.append({"method": name, **hits(r, t_), **{f"verb_{k}": v for k, v in hits(verb_recs, tv).items()}})

    # ---- 4. references ----
    rates = R.global_rates(train.refs)
    (out_dir / "ref_global_rates.json").write_text(json.dumps({r: v for r, v in rates.items() if v >= 1e-5}))
    has_ref = (queries.refs.str.len() > 0).values
    rtruth = [set(x) for x in queries.refs]
    gref = [r for r, _ in sorted(rates.items(), key=lambda x: -x[1])[:5]]
    ref_methods = {"Global most cited": [gref] * len(queries),
                   "Retrieval + count": [[r for r, *_ in R.recommend(list(docs.refs.values[ix]), s, rates, "count")]
                                         for ix, s in found],
                   "Retrieval + specificity (log-lift)": [[r for r, *_ in R.recommend(list(docs.refs.values[ix]), s, rates)]
                                                          for ix, s in found]}
    # direct multi-label classifier on the 300 most cited references
    top_refs = [r for r, _ in sorted(rates.items(), key=lambda x: -x[1])[:300]]
    samp = train.sample(150_000, random_state=cfg["seed"])
    mlb = MultiLabelBinarizer(classes=top_refs)
    y = mlb.fit_transform([[r for r in x if r in set(top_refs)] for x in samp.refs])
    t = time.perf_counter()
    vecz = make_vectorizer(cfg)
    clf = OneVsRestClassifier(SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=15, tol=1e-3,
                                            random_state=cfg["seed"]), n_jobs=-1)
    clf.fit(vecz.fit_transform(samp.problem_masked), y)
    proba = clf.predict_proba(vecz.transform(queries.problem_masked))
    ref_methods["Direct classifier (TF-IDF + one-vs-rest LR, 300 refs)"] = [
        [top_refs[j] for j in np.argsort(-p)[:5]] for p in proba]
    log.info("reference classifier trained in %.0f s", time.perf_counter() - t)
    generic = gref[0]
    ref_rows = []
    for name, recs in ref_methods.items():
        r = [recs[i] for i in np.where(has_ref)[0]]
        t_ = [rtruth[i] for i in np.where(has_ref)[0]]
        ref_rows.append({"method": name, **hits(r, t_),
                         f"top-1 is {generic} (%)": round(100 * float(np.mean([bool(x) and x[0] == generic for x in r])), 1)})

    # ---- example ----
    ex = []
    for i in rng.choice(np.where(has_act & has_ref)[0], 3, replace=False):
        ix, s = found[i]
        ex.append(f"- **{queries.problem_masked[i][:120]}** ({queries.family[i]})\n"
                  f"  - actually done: {', '.join(queries.actions[i])}; cited: {', '.join(queries.refs[i][:4])}\n"
                  f"  - recommended actions: " + ", ".join(f"{a} ({p:.0%})" for a, p in aggregate(list(docs.actions.values[ix]), s, 3)) +
                  "\n  - recommended references: " + ", ".join(r for r, *_ in R.recommend(list(docs.refs.values[ix]), s, rates, top=3)) + "\n")

    md = ["# Recommenders (Phase 6)\n\n",
          "Generated by `python -m aerofix.recommend.evaluate`. 2,000 test (2025) queries; the Phase 5 retriever "
          f"(hybrid BM25 + fine-tuned MiniLM + cross-encoder rerank, {ms:.0f} ms/query) returns the top {K} past cases "
          "(train + val, same family).\n\n",
          f"## Canonical actions\n\nRules extract fix actions (verb + last two object words) from the Phase 3 action "
          f"text; {100 * n_with:.1f}% of reports yield at least one. {len(mapping)} frequent object strings were merged "
          "by MiniLM embedding + agglomerative clustering (cosine distance < 0.15). Largest merges:\n\n",
          md_table(merges, index=False), "\n\n",
          f"## Corrective-action recommendation ({int(has_act.sum())} test queries with an action)\n\n"
          "hit@k: a recommended canonical action (verb + object) appears in what was actually done; verb_hit@k: "
          "the verb alone matches.\n\n", md_table(pd.DataFrame(act_rows), index=False), "\n\n",
          f"## Manual-reference recommendation ({int(has_ref.sum())} test queries citing a reference)\n\n"
          "hit@k: a recommended reference was actually cited in the report. The last column shows how often the "
          "top recommendation is the single most generic reference.\n\n", md_table(pd.DataFrame(ref_rows), index=False),
          "\n\n## Examples\n\n", *ex]
    (cfg["paths"]["reports"] / "recommend.md").write_text("".join(md), encoding="utf-8")
    log.info("wrote reports/recommend.md\n%s\n%s", pd.DataFrame(act_rows).to_string(), pd.DataFrame(ref_rows).to_string())


if __name__ == "__main__":
    main()
