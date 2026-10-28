"""Cited summary with the local LLM + citation validation.

The LLM gets ONLY assembled evidence: aircraft, fault, predicted system, the
recommended actions and references with their frequencies, the top similar past
cases (labelled [Case 1] ...) and FAA passages (labelled [AC 43.13-1B p.212] ...).
It must cite those labels, must not write step-by-step procedures, and must point
to the approved data ("refer to AMM 52-10-00") instead. Temperature is low (0.2)
so the summary sticks to the evidence.

The validator then checks the output mechanically:
  * every [label] cited must exist in the context (otherwise: invented citation)
  * every manual reference written in the summary (AMM 52-10-00 ...) must appear in
    the context (otherwise: hallucinated reference)
  * at least one citation must be present
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import faiss
import numpy as np
import pandas as pd

from aerofix.config import load_config
from aerofix.preprocess.refs import extract_refs
from aerofix.rag.ata import title

SYSTEM = """You are AeroFix, a decision-support assistant for airline line maintenance technicians.
Write a short summary (3 sentences) of what the evidence suggests for the reported fault.
The confidence percentages belong to the predicted ATA system and part condition only; quote them that way.
Rules:
- Use ONLY the evidence given. EVERY sentence must end with at least one citation label in square brackets, copied exactly from the evidence, e.g. [Case 2] or [AC 43.13-1B p.212]. Actions and references list the cases that support them: cite those. Use at most 2 labels per sentence.
- Mention the most likely system, the most common corrective actions in similar cases (with how often), and the manual references technicians used.
- Do NOT write step-by-step procedures, torque values, limits or part numbers that are not in the evidence. For any procedure say "refer to <reference> in the approved maintenance data".
- Do NOT invent references or case numbers. If the evidence is weak or mixed, say so.
- Plain text, no headings, no bullet points.
Example of the expected style (for a different fault):
"The fault most likely belongs to ATA 33 LIGHTS (95% confidence) [Case 1]. In similar cases technicians most often replaced the battery pack (40%) [Case 1] [Case 3]; refer to AMM 33-51-06 in the approved maintenance data [Case 3]. General FAA guidance on battery installation is in [AMT Airframe p.443]." """


@dataclass
class Evidence:
    family: str
    variant: str
    problem: str
    ata2: str
    ata2_conf: float
    ata4: str
    condition: str
    condition_conf: float
    actions: list[tuple[str, float]]       # (action, vote share)
    refs: list[tuple[str, float]]          # (reference, support share)
    cases: list[dict]                      # problem, action, refs, year, variant
    passages: list[dict] = field(default_factory=list)  # label, heading, text

    def labels(self) -> set[str]:
        return {f"Case {i + 1}" for i in range(len(self.cases))} | {p["label"] for p in self.passages}

    def context_refs(self) -> set[str]:
        out = {r for r, _ in self.refs}
        for c in self.cases:
            out |= set(c.get("refs", []))
        return out

    def _support(self, key: str, item: str) -> str:
        """Labels of the shown cases that contain an action / reference, so the LLM can cite them."""
        ids = [f"[Case {i + 1}]" for i, c in enumerate(self.cases) if item in c.get(key, [])]
        return f"; see {' '.join(ids)}" if ids else ""

    def render(self) -> str:
        w = lambda s, n: " ".join(str(s).split()[:n])
        lines = [f"AIRCRAFT: {self.family} family, {self.variant}",
                 f"FAULT REPORTED: {self.problem}",
                 f"PREDICTED SYSTEM: ATA {self.ata2} {title(self.ata2)} ({self.ata2_conf:.0%} confidence), "
                 f"JASC {self.ata4}; likely part condition: {self.condition} ({self.condition_conf:.0%})",
                 "MOST COMMON CORRECTIVE ACTIONS IN SIMILAR CASES: " +
                 ("; ".join(f"{a} ({p:.0%}{self._support('actions', a)})" for a, p in self.actions)
                  or "none extracted"),
                 "MANUAL REFERENCES TECHNICIANS CITED IN SIMILAR CASES: " +
                 ("; ".join(f"{r} (in {s:.0%} of similar cases{self._support('refs', r)})" for r, s in self.refs)
                  or "none"),
                 "", "SIMILAR PAST CASES:"]
        for i, c in enumerate(self.cases, 1):
            lines.append(f"[Case {i}] ({c['variant']}, {c['year']}) PROBLEM: {w(c['problem'], 30)} | "
                         f"ACTION: {w(c['action'], 25) or 'not recorded'}" +
                         (f" | REFS: {', '.join(c['refs'][:3])}" if c.get("refs") else ""))
        if self.passages:
            lines += ["", "FAA PUBLIC REFERENCE PASSAGES (general guidance, not aircraft-specific):"]
            for p in self.passages:
                lines.append(f"[{p['label']}] {p['heading']}: {w(p['text'], 80)}")
        # The mask token "[REF]" looks like a citation label to the LLM (it cited it): show it as <ref>.
        return "\n".join(lines).replace("[REF]", "<ref>")


class PassageIndex:
    def __init__(self, cfg: dict | None = None):
        from sentence_transformers import SentenceTransformer
        cfg = cfg or load_config()
        d = cfg["paths"]["models"] / "rag"
        self.df = pd.read_parquet(d / "passages.parquet")
        self.index = faiss.read_index(str(d / "passages.faiss"))
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, problem: str, ata2: str, k: int = 3) -> list[dict]:
        """Query = chapter title + fault text, so 'DAMPER LEAKING' is read in the context of DOORS."""
        from aerofix.retrieve.train import encode
        _, nn = self.index.search(encode(self.enc, [f"{title(ata2)}: {problem}"]), k)
        return [self.df.iloc[i][["label", "heading", "text"]].to_dict() for i in nn[0] if i >= 0]


def trim_to_sentence(text: str) -> str:
    """The token cap can cut the last sentence; drop an unfinished tail rather than show half a claim."""
    text = text.strip()
    if text.endswith((".", "]")):
        return text
    cut = max(text.rfind(". "), text.rfind("]. "), text.rfind("] "))
    return text[:cut + 1].strip() if cut > 0 else text


def validate(summary: str, ev: Evidence) -> dict:
    cited = re.findall(r"\[([^\[\]]+)\]", summary)
    allowed = ev.labels()
    invalid = [c for c in cited if c.strip() not in allowed]
    refs_written = extract_refs(summary.upper())
    known = ev.context_refs()
    hallucinated = [r for r in refs_written if r not in known]
    return {"n_citations": len(cited), "invalid_citations": invalid, "hallucinated_refs": hallucinated,
            "has_citation": bool(cited), "ok": bool(cited) and not invalid and not hallucinated}


def generate_stream(ev: Evidence, cfg: dict | None = None, model: str | None = None):
    """Yield the summary token by token (for the app); same prompt and options as generate()."""
    import ollama
    cfg = cfg or load_config()
    client = ollama.Client(host=cfg["llm"]["host"])
    for chunk in client.chat(model=model or cfg["llm"]["model"], stream=True,
                             messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ev.render()}],
                             options={"temperature": 0.2, "num_predict": 170, "num_ctx": 4096, "seed": 42}, keep_alive="30m"):
        yield chunk["message"]["content"]


def generate(ev: Evidence, cfg: dict | None = None, model: str | None = None) -> dict:
    import ollama
    cfg = cfg or load_config()
    client = ollama.Client(host=cfg["llm"]["host"])
    t = time.perf_counter()
    for m in ([model] if model else [cfg["llm"]["model"], cfg["llm"]["fallback_model"]]):
        try:
            r = client.chat(model=m, messages=[{"role": "system", "content": SYSTEM},
                                               {"role": "user", "content": ev.render()}],
                            options={"temperature": 0.2, "num_predict": 170, "num_ctx": 4096, "seed": 42}, keep_alive="30m")
            summary = trim_to_sentence(r["message"]["content"])
            return {"summary": summary, "model": m, "seconds": round(time.perf_counter() - t, 2),
                    **validate(summary, ev)}
        except Exception as e:  # model missing / Ollama down -> try fallback
            last = e
    return {"summary": "", "model": None, "seconds": round(time.perf_counter() - t, 2), "error": str(last),
            "n_citations": 0, "invalid_citations": [], "hallucinated_refs": [], "has_citation": False, "ok": False}
