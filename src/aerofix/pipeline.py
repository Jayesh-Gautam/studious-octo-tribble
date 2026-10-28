"""End-to-end advisor: fault text in, all AeroFix outputs out (used by the RAG
evaluation, the FastAPI service and the Streamlit app).

    adv = Advisor()                       # loads every model once (~30 s)
    out = adv.advise("A320", "A321", "L1 DOOR DAMPER LEAKING, EMER CYL PRESS LOW")

Steps (each timed):
  1. preprocess   clean, strip boilerplate, mask manual references, normalise (lemma)
  2. classify     multi-task DistilBERT -> ATA chapter, JASC code, part condition
  3. retrieve     hybrid BM25 + fine-tuned MiniLM + cross-encoder, same family, top 20
  4. recommend    similarity-weighted actions; references by specificity (log-lift)
  5. passages     public FAA document passages for the predicted system
  6. summary      local LLM, cited, validated (optional: llm=False skips it)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from aerofix.classify.data import TARGETS, LabelSpace
from aerofix.classify.transformer import MultiTaskTransformer
from aerofix.config import load_config
from aerofix.preprocess.clean import basic_clean, strip_boilerplate
from aerofix.preprocess.normalise import Normaliser
from aerofix.preprocess.refs import mask_refs
from aerofix.rag.ata import title
from aerofix.rag.generate import Evidence, PassageIndex, generate
from aerofix.recommend import references as R
from aerofix.recommend.actions import aggregate
from aerofix.retrieve.search import Retriever

DISCLAIMER = ("Decision support only. AeroFix summarises public FAA Service Difficulty Reports and FAA guidance; "
              "it does not contain OEM procedures. Always follow the approved AMM/SRM/FIM and your organisation's "
              "procedures.")


class Advisor:
    def __init__(self, cfg: dict | None = None, include_test: bool = False, k: int = 20):
        self.cfg = cfg = cfg or load_config()
        self.k = k
        m = cfg["paths"]["models"]
        proc = cfg["paths"]["processed"]
        self.norm = Normaliser.load(m / "normaliser.pkl")
        self.labels = LabelSpace.load(m / "classify" / "labels.json")
        best = json.loads((m / "classify" / "best.json").read_text())["best"]
        cdir = m / "classify" / best
        meta = json.loads((cdir / "meta.json").read_text())
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(cdir / "tokenizer"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clf = MultiTaskTransformer(meta["model"], {t: len(self.labels.names(t)) for t in TARGETS})
        self.clf.load_state_dict(torch.load(cdir / "model.pt", map_location="cpu"))
        self.clf.to(self.device).eval()
        self.max_len = meta["max_seq_len"]

        df = pd.read_parquet(proc / "sdr_a320_b737.parquet",
                             columns=["row_id", "split", "family", "variant", "file_year", "ata4", "refs", "text"])
        df = (df.merge(pd.read_parquet(proc / "sdr_segments.parquet", columns=["row_id", "problem_masked", "action_text"]), on="row_id")
                .merge(pd.read_parquet(proc / "sdr_problem_norm.parquet", columns=["row_id", "problem_lemma"]), on="row_id")
                .merge(pd.read_parquet(proc / "sdr_actions.parquet"), on="row_id"))
        keep = df.split.isin(["train", "val", "test"] if include_test else ["train", "val"]).values
        vec = np.load(m / "retrieve" / "vectors_ft.npy")[df.row_id.values[keep]]
        self.docs = df[keep].reset_index(drop=True)
        self.retriever = Retriever(self.docs, vec, cfg)
        self.rates = json.loads((m / "recommend" / "ref_global_rates.json").read_text())
        self.passages = PassageIndex(cfg)
        self.warm_up()

    def warm_up(self) -> None:
        """Pay one-off costs at load time, not on the user's first query: WordNet + POS tagger
        loading (~3.5 s), CUDA kernel set-up, and loading the LLM into GPU memory (kept 30 min)."""
        self.advise("A320", "A320", "LAV SMOKE DETECTOR INOP", llm=False)
        try:
            import ollama
            ollama.Client(host=self.cfg["llm"]["host"]).generate(model=self.cfg["llm"]["model"], prompt="",
                                                                  keep_alive="30m")
        except Exception:  # Ollama not running: the app still works without the summary
            pass

    @torch.no_grad()
    def classify(self, problem: str) -> dict:
        enc = self.tok([problem.lower()], truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        logits = self.clf(enc["input_ids"], enc["attention_mask"])
        out = {}
        for t in TARGETS:
            p = torch.softmax(logits[t].float(), -1)[0].cpu().numpy()
            top = np.argsort(-p)[:3]
            out[t] = [(self.labels.names(t)[i], float(p[i])) for i in top]
        return out

    def advise(self, family: str, variant: str, text: str, stage: str | None = None, llm: bool = True) -> dict:
        times, t0 = {}, time.perf_counter()

        def tick(name, t):
            times[name] = round(1000 * (time.perf_counter() - t), 1)
            return time.perf_counter()

        t = time.perf_counter()
        clean, context = strip_boilerplate(basic_clean(text))
        problem = mask_refs(clean, self.cfg["preprocess"]["mask_token"])
        normalised = self.norm.normalise(problem, morph="none")
        lemma = self.norm.normalise(problem, morph="lemma")
        t = tick("preprocess_ms", t)

        pred = self.classify(problem)
        t = tick("classify_ms", t)

        idx, sims = self.retriever.search(problem, lemma, family, k=self.k)
        cases = self.docs.iloc[idx]
        t = tick("retrieve_ms", t)

        actions = aggregate(list(cases.actions), sims, top=5)
        refs = R.recommend(list(cases.refs), sims, self.rates, "specificity", top=5)
        t = tick("recommend_ms", t)

        ata2 = pred["ata2"][0][0]
        passages = self.passages.search(problem, ata2, k=2)  # 2 passages keep the LLM prompt (and latency) small
        t = tick("passages_ms", t)

        evidence_cases = [{"problem": c.problem_masked, "action": c.action_text, "refs": list(c.refs),
                           "actions": list(c.actions),
                           "year": int(c.file_year), "variant": c.variant, "row_id": int(c.row_id),
                           "similarity": float(s), "text": c.text} for c, s in zip(cases.itertuples(), sims)]
        ev = Evidence(family=family, variant=variant, problem=problem, ata2=ata2, ata2_conf=pred["ata2"][0][1],
                      ata4=pred["ata4"][0][0], condition=pred["condition"][0][0],
                      condition_conf=pred["condition"][0][1], actions=actions,
                      refs=[(r, s) for r, _, s in refs], cases=evidence_cases[:5], passages=passages)
        summary = generate(ev, self.cfg) if llm else None
        if llm:
            tick("llm_ms", t)
        times["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return {
            "input": {"family": family, "variant": variant, "stage": stage, "text": text},
            "under_the_hood": {"cleaned": clean, "context_prefix": context, "masked": problem,
                               "normalised": normalised, "lemmatised": lemma},
            "system": {"ata2": [(c, p, title(c)) for c, p in pred["ata2"]], "ata4": pred["ata4"]},
            "condition": pred["condition"],
            "actions": actions,
            "references": [{"reference": r, "score": round(sc, 4), "cited_in_share": round(s, 3),
                            "global_share": round(self.rates.get(r, 0.0), 4)} for r, sc, s in refs],
            "cases": evidence_cases[:5],
            "passages": passages,
            "summary": summary,
            "disclaimer": DISCLAIMER,
            "timings": times,
            "evidence": ev,
        }
