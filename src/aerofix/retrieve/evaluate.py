"""Train + evaluate retrieval: python -m aerofix.retrieve.evaluate [--judge 50]

Stage "val":  1,000 val (2024) queries searched in the TRAIN index  -> choose BM25 variant
Stage "test": 2,000 test (2025) queries searched in TRAIN+VAL      -> reported numbers
Search is always restricted to the query's aircraft family.

Methods: BM25 (raw / normalised / stem / lemma), average Word2Vec, MiniLM zero-shot,
MiniLM fine-tuned, hybrid (RRF of best BM25 + fine-tuned MiniLM), and cross-encoder
re-ranking of the top 50. Artefacts in models/retrieve/: biencoder/, crossencoder/,
vectors_ft.npy (fine-tuned embedding of every report, row order = row_id).
--judge N: the local LLM judges the top 5 of N test queries for 3 methods (silver
check of the proxy relevance labels).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

from aerofix.config import load_config
from aerofix.preprocess.tokenize import tokenize
from aerofix.retrieve.common import gains, load_corpus, metrics_for, rrf
from aerofix.retrieve.index import BM25, DenseIndex, normalise_rows
from aerofix.retrieve.train import CrossEncoder, encode, mine_triples, train_biencoder
from aerofix.utils import get_logger, md_table, set_seed

log = get_logger(__name__)
TOP = 50
BM25_COLS = {"BM25 raw (tokenised)": "problem_tok", "BM25 + abbrev + spelling": "problem_norm",
             "BM25 + stem": "problem_stem", "BM25 + lemma": "problem_lemma"}


def w2v_vectors(w2v, texts) -> np.ndarray:
    out = np.zeros((len(texts), w2v.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vs = [w2v.wv[w] for w in t.split() if w in w2v.wv]
        if vs:
            out[i] = np.mean(vs, axis=0)
    return normalise_rows(out)


def score_rankings(rankings: dict[str, list[np.ndarray]], queries: pd.DataFrame, docs: pd.DataFrame,
                   timing: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, ranks in rankings.items():
        m = [metrics_for(gains(q, docs.iloc[r[:10]])) for (_, q), r in zip(queries.iterrows(), ranks)]
        rows.append({"method": name, **{k: round(float(np.mean([x[k] for x in m])), 4) for k in m[0]},
                     "ms_per_query": round(timing.get(name, float("nan")), 1)})
    return pd.DataFrame(rows).sort_values("NDCG@5", ascending=False)


def run_stage(name, docs, queries, vecs, cfg, cross=None, bm25_choice=None):
    """docs/queries: DataFrames; vecs: {model: (doc_vectors, query_vectors)}. Returns (table, rankings)."""
    rankings, timing = {}, {}
    qfam = queries.family.values
    for label, col in BM25_COLS.items():  # all variants are scored (normalisation ablation)
        bm = BM25().fit(list(docs[col]))
        t = time.perf_counter()
        rankings[label] = [bm.search(q, TOP, mask=(docs.family.values == f))
                           for q, f in zip(queries[col], qfam)]
        timing[label] = 1000 * (time.perf_counter() - t) / len(queries)
        log.info("%s %s done", name, label)
    for label, (dv, qv) in vecs.items():
        idx = DenseIndex(dv, docs.family.values)
        t = time.perf_counter()
        rankings[label] = idx.search(qv, qfam, TOP)
        timing[label] = 1000 * (time.perf_counter() - t) / len(queries)
    best_bm25 = bm25_choice or max((k for k in rankings if k.startswith("BM25")),
                                   key=lambda k: np.mean([metrics_for(gains(q, docs.iloc[r[:10]]))["NDCG@5"]
                                                          for (_, q), r in zip(queries.iterrows(), rankings[k])]))
    ft = "MiniLM fine-tuned"
    rankings[f"Hybrid RRF ({best_bm25} + {ft})"] = [rrf([a, b]) for a, b in zip(rankings[best_bm25], rankings[ft])]
    timing[f"Hybrid RRF ({best_bm25} + {ft})"] = timing[best_bm25] + timing[ft]
    if cross is not None:
        for base in (ft, f"Hybrid RRF ({best_bm25} + {ft})"):
            t = time.perf_counter()
            reranked = []
            for q, r in zip(queries.problem_masked, rankings[base]):
                s = cross.score(q, list(docs.problem_masked.values[r]))
                reranked.append(r[np.argsort(-s)])
            rankings[f"{base} + cross-encoder rerank"] = reranked
            timing[f"{base} + cross-encoder rerank"] = timing[base] + 1000 * (time.perf_counter() - t) / len(queries)
    return score_rankings(rankings, queries, docs, timing), rankings, best_bm25


def llm_judge(queries, docs, rankings, methods, n, cfg) -> pd.DataFrame:
    import ollama
    client = ollama.Client(host=cfg["llm"]["host"])
    schema = {"type": "object", "properties": {"relevant": {"type": "boolean"}}, "required": ["relevant"]}
    system = ("You compare two aircraft maintenance fault descriptions from the same aircraft family. Answer "
              "relevant=true if the PAST report describes the same kind of fault: the same system or part with a "
              "similar condition (e.g. both a cracked frame, both an inoperative emergency light battery). "
              "Different part or different failure -> false.")
    cache: dict[tuple[int, int], bool] = {}
    rows = []
    for m in methods:
        hits = []
        for qi in range(n):
            q = queries.iloc[qi]
            for d in rankings[m][qi][:5]:
                key = (qi, int(d))
                if key not in cache:
                    msg = f"NEW: {q.problem_masked}\nPAST: {docs.problem_masked.values[d]}"
                    r = client.chat(model=cfg["llm"]["model"], format=schema, options={"temperature": 0, "seed": 42},
                                    messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
                    cache[key] = bool(json.loads(r["message"]["content"])["relevant"])
                hits.append(cache[key])
        rows.append({"method": m, "LLM-judged P@5": round(float(np.mean(hits)), 3)})
    # agreement between proxy label and LLM verdict over all judged pairs
    proxy = [gains(queries.iloc[qi], docs.iloc[[d]])[0] == 2 for (qi, d) in cache]
    agree = np.mean([p == v for p, v in zip(proxy, cache.values())])
    return pd.DataFrame(rows), float(agree), len(cache)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=int, default=50)
    ap.add_argument("--triples", type=int, default=100_000)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    rdir = cfg["paths"]["models"] / "retrieve"
    rdir.mkdir(parents=True, exist_ok=True)
    df = load_corpus(cfg)
    df["problem_tok"] = [" ".join(tokenize(t)) for t in df.problem_masked]
    assert (df.row_id.values == np.arange(len(df))).all()

    # ---- bi-encoder: zero-shot and fine-tuned ----
    zs = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t = time.perf_counter()
    vec_zs = encode(zs, list(df.problem_masked))
    log.info("encoded corpus (zero-shot) in %.0f s", time.perf_counter() - t)
    del zs
    if (rdir / "biencoder").exists():
        ft = SentenceTransformer(str(rdir / "biencoder"))
    else:
        triples = mine_triples(df[df.split == "train"], args.triples, cfg["seed"])
        log.info("mined %d triples", len(triples))
        ft = train_biencoder(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), triples, seed=cfg["seed"])
        ft.save(str(rdir / "biencoder"))
    vec_ft = encode(ft, list(df.problem_masked))
    np.save(rdir / "vectors_ft.npy", vec_ft.astype(np.float16))
    w2v = Word2Vec.load(str(cfg["paths"]["models"] / "classify" / "bilstm" / "word2vec.model"))
    vec_w2v = w2v_vectors(w2v, list(df.problem_norm))

    # ---- cross-encoder trained on fine-tuned bi-encoder candidates of TRAIN queries ----
    train_idx = np.where(df.split == "train")[0]
    if (rdir / "crossencoder").exists():
        cross = CrossEncoder(str(rdir / "crossencoder"))
    else:
        rng = np.random.default_rng(cfg["seed"])
        qs = rng.choice(train_idx, 6000, replace=False)
        index = DenseIndex(vec_ft[train_idx], df.family.values[train_idx])
        cands = index.search(vec_ft[qs], df.family.values[qs], 21)
        qtext, dtext, lab = [], [], []
        for q, c in zip(qs, cands):
            c = train_idx[c]
            c = c[c != q][:20]
            g = gains(df.iloc[q], df.iloc[c])
            qtext += [df.problem_masked.values[q]] * len(c)
            dtext += list(df.problem_masked.values[c])
            lab += list((g == 2).astype(float))
        log.info("cross-encoder pairs: %d (positive %.1f%%)", len(lab), 100 * np.mean(lab))
        cross = CrossEncoder().fit(qtext, dtext, np.array(lab), seed=cfg["seed"])
        cross.model.save_pretrained(rdir / "crossencoder")
        cross.tok.save_pretrained(rdir / "crossencoder")

    rng = np.random.default_rng(cfg["seed"])
    results, chosen = {}, None
    for stage, doc_mask, qsplit, nq in (("val", df.split == "train", "val", 1000),
                                        ("test", df.split.isin(["train", "val"]), "test", 2000)):
        didx = np.where(doc_mask)[0]
        qidx = rng.choice(np.where(df.split == qsplit)[0], nq, replace=False)
        docs, queries = df.iloc[didx].reset_index(drop=True), df.iloc[qidx].reset_index(drop=True)
        vecs = {"Average Word2Vec": (vec_w2v[didx], vec_w2v[qidx]),
                "MiniLM zero-shot": (vec_zs[didx], vec_zs[qidx]),
                "MiniLM fine-tuned": (vec_ft[didx], vec_ft[qidx])}
        # The hybrid's BM25 variant is chosen on val only; on test it is fixed to that choice.
        table, rankings, best = run_stage(stage, docs, queries, vecs, cfg, cross=cross, bm25_choice=chosen)
        chosen = chosen or best
        results[stage] = table
        log.info("%s:\n%s", stage, table.to_string(index=False))
        if stage == "test" and args.judge:
            methods = [chosen, "MiniLM fine-tuned", f"Hybrid RRF ({chosen} + MiniLM fine-tuned) + cross-encoder rerank"]
            methods = [m for m in methods if m in rankings]
            judged, agree, n_pairs = llm_judge(queries, docs, rankings, methods, args.judge, cfg)
            results["judge"] = (judged, agree, n_pairs)
        torch.cuda.empty_cache()

    md = ["# Retrieval of similar past cases (Phase 5)\n\n",
          "Generated by `python -m aerofix.retrieve.evaluate`. Index text and queries: `problem_masked`; search is "
          "restricted to the query's aircraft family. **Proxy relevance**: same JASC code (ata4) + same part "
          "condition. Hit@k = at least one relevant report in the top k; NDCG@5 gives 2 points for ata4 + condition, "
          "1 for ata4 only.\n\n",
          f"## Val (1,000 2024 queries, train index) - used to choose the BM25 variant: **{chosen}**\n\n",
          md_table(results["val"], index=False), "\n\n",
          "## Test (2,000 2025 queries, train + val index)\n\n", md_table(results["test"], index=False), "\n"]
    if "judge" in results:
        judged, agree, n_pairs = results["judge"]
        md += [f"\n## LLM sanity check (silver)\n\nThe local LLM ({cfg['llm']['model']}) judged whether each of the "
               f"top 5 results describes the same kind of fault, for {args.judge} test queries ({n_pairs} distinct "
               f"pairs). Agreement between LLM verdict and proxy label: **{agree:.1%}**.\n\n",
               md_table(judged, index=False), "\n"]
    # The app uses the method that was best on VAL (NDCG@5), whatever test says afterwards.
    best = results["val"].iloc[0].method
    (rdir / "best.json").write_text(json.dumps({"best": best, "bm25_variant": chosen}))
    md.append(f"\nSelected for the app (best val NDCG@5): **{best}**.\n")
    (cfg["paths"]["reports"] / "retrieval.md").write_text("".join(md), encoding="utf-8")
    log.info("wrote reports/retrieval.md")


if __name__ == "__main__":
    main()
