"""AeroFix HTTP service + web console: uvicorn aerofix.app.api:app --port 8000

Serves the JSON API under /api and, if it has been built (frontend/dist), the
React + shadcn/ui console at /. The Advisor (all models) is loaded once at start.

  POST /api/advise                 structured advice (fast); returns request_id
  GET  /api/summary/{request_id}   cited LLM summary, streamed as server-sent events
  GET  /api/documents/{doc}        the full public FAA PDF (open at #page=N)
  GET  /api/documents/{doc}/pages/{n}   just pages n-1..n+1 as a small PDF (fast viewer)
  GET  /api/cases/{row_id}         the full original SDR report
  GET  /api/meta | /api/metrics | /api/dataset | /api/feedback | /api/health
  POST /api/feedback
The old un-prefixed /advise, /feedback and /health routes still work.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aerofix.config import load_config
from aerofix.feedback import store
from aerofix.pipeline import Advisor
from aerofix.rag.ata import ATA_TITLES, title
from aerofix.rag.docs import SHORT_NAMES
from aerofix.rag.generate import generate_stream, trim_to_sentence, validate

STATE: dict = {}
EVIDENCE: OrderedDict = OrderedDict()  # request_id -> Evidence, for streaming the summary afterwards
DOC_FILES = {short: fname for fname, short in SHORT_NAMES.items()}  # whitelist: only these PDFs are served
STAGES = {"IN": "Inspection / maintenance", "TX": "Taxi", "TO": "Takeoff", "CL": "Climb", "CR": "Cruise",
          "DE": "Descent", "AP": "Approach", "LD": "Landing"}
EXAMPLES = ["L1 door damper leaking, emer cyl press low", "FWD lav smoke detector inop",
            "Corrosion found on frame 16 lower flange at floor beam",
            "Emergency exit light battery pack failed capacity test", "Aft cargo door will not latch"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    STATE["cfg"] = cfg
    STATE["advisor"] = Advisor(cfg)
    proc = cfg["paths"]["processed"]
    reports = pd.read_parquet(proc / "sdr_a320_b737.parquet",
                              columns=["row_id", "split", "family", "variant", "file_year", "difficulty_date", "ata2",
                                       "ata4", "PartCondition", "PartName", "PartLocation", "StageOfOperationCode",
                                       "Discrepancy", "refs"])
    STATE["reports"] = reports.set_index("row_id")
    yield
    STATE.clear()


app = FastAPI(title="AeroFix", description="Maintenance fault advisor (decision support only)", lifespan=lifespan)
router = APIRouter()


class AdviseRequest(BaseModel):
    family: str = Field(pattern="^(A320|B737)$")
    variant: str = ""
    text: str = Field(min_length=3, max_length=2000)
    stage: str | None = None
    llm: bool = False  # the console streams the summary separately via /api/summary


class FeedbackRequest(BaseModel):
    text: str
    family: str | None = None
    variant: str | None = None
    stage: str | None = None
    predicted_ata2: str | None = None
    predicted_condition: str | None = None
    system_correct: bool | None = None
    correct_ata2: str | None = None
    action_fixed: str | None = None
    comment: str | None = None


def _plain(x):
    """numpy scalars / tuples / timestamps -> JSON-friendly Python types."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else x.date().isoformat()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _ollama_ok() -> bool:
    try:
        import ollama
        ollama.Client(host=STATE["cfg"]["llm"]["host"]).list()
        return True
    except Exception:
        return False


@router.get("/health")
def health():
    cfg = STATE.get("cfg") or load_config()
    return {"status": "ok", "models_loaded": "advisor" in STATE, "llm_online": _ollama_ok() if STATE else False,
            "llm_model": cfg["llm"]["model"]}


@router.get("/meta")
def meta():
    reports = STATE["reports"]
    variants = {f: g.variant.value_counts().index.tolist() for f, g in reports.groupby("family")}
    return {"families": ["A320", "B737"], "variants": variants, "stages": STAGES, "examples": EXAMPLES,
            "ata_titles": ATA_TITLES, "documents": list(DOC_FILES)}


@router.post("/advise")
def advise(req: AdviseRequest):
    if "advisor" not in STATE:
        raise HTTPException(503, "models still loading")
    out = STATE["advisor"].advise(req.family, req.variant, req.text, stage=req.stage, llm=req.llm)
    rid = uuid4().hex
    EVIDENCE[rid] = out.pop("evidence")
    while len(EVIDENCE) > 100:  # bounded memory
        EVIDENCE.popitem(last=False)
    out["request_id"] = rid
    return _plain(out)


@router.get("/summary/{request_id}")
def summary(request_id: str):
    ev = EVIDENCE.get(request_id)
    if ev is None:
        raise HTTPException(404, "unknown request_id (run /advise first)")

    def events():
        text = ""
        try:
            for tok in generate_stream(ev, STATE["cfg"]):
                text += tok
                yield f"data: {json.dumps({'token': tok})}\n\n"
        except Exception as e:  # Ollama not running
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return
        final = trim_to_sentence(text)
        yield f"event: done\ndata: {json.dumps({'text': final, **validate(final, ev)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _doc_path(doc: str) -> Path:
    if doc not in DOC_FILES:
        raise HTTPException(404, f"unknown document {doc!r}")
    p = STATE["cfg"]["paths"]["raw_docs"] / DOC_FILES[doc]
    if not p.exists():
        raise HTTPException(404, "document file missing (run download_data.py)")
    return p


@router.get("/documents/{doc}")
def document(doc: str):
    return FileResponse(_doc_path(doc), media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{DOC_FILES[doc]}"'})


@lru_cache(maxsize=64)
def _page_excerpt(doc: str, page: int) -> bytes:
    import pymupdf
    src = pymupdf.open(_doc_path(doc))
    first, last = max(1, page - 1), min(src.page_count, page + 1)
    out = pymupdf.open()
    out.insert_pdf(src, from_page=first - 1, to_page=last - 1)
    out.set_metadata({"title": f"{doc}, pages {first}-{last} (cited page {page})"})
    return out.tobytes()


@router.get("/documents/{doc}/pages/{page}")
def document_pages(doc: str, page: int):
    """The cited page with one page of context either side (small, opens instantly)."""
    return Response(_page_excerpt(doc, page), media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{doc} p{page}.pdf"'})


@router.get("/cases/{row_id}")
def case(row_id: int):
    reports = STATE["reports"]
    if row_id not in reports.index:
        raise HTTPException(404, "unknown report")
    r = reports.loc[row_id]
    return _plain({"row_id": row_id, "family": r.family, "variant": r.variant, "year": int(r.file_year),
                   "date": r.difficulty_date, "ata2": r.ata2, "ata_title": title(r.ata2), "jasc": r.ata4,
                   "condition": r.PartCondition, "part_name": r.PartName, "part_location": r.PartLocation,
                   "stage": STAGES.get(r.StageOfOperationCode, r.StageOfOperationCode), "refs": list(r.refs),
                   "report": r.Discrepancy, "split": r.split})


@router.get("/metrics")
def metrics():
    return _metrics()


@lru_cache(maxsize=1)
def _metrics():
    from aerofix.eval.final import table, test_ablations
    cfg = STATE["cfg"]
    rep, mdl = cfg["paths"]["reports"], cfg["paths"]["models"]
    cls = pd.read_csv(rep / "classification_results.csv")
    leak, fam = test_ablations(cfg)
    lc = pd.read_csv(mdl / "feedback_loop" / "learning_curve.csv")
    rec = lambda h: table(rep / "recommend.md", h).to_dict("records")
    return _plain({
        "classification": cls.to_dict("records"),
        "leakage": leak.to_dict("records"), "cross_family": fam.to_dict("records"),
        "retrieval": table(rep / "retrieval.md", "Test (").to_dict("records"),
        "retrieval_llm_check": table(rep / "retrieval.md", "LLM sanity").to_dict("records"),
        "actions": rec("Corrective-action"), "references": rec("Manual-reference"),
        "segmentation": table(rep / "segmentation.md", "Results").to_dict("records"),
        "language_model": table(rep / "normalisation.md", "2. N-gram").to_dict("records"),
        "rag": {"qwen": table(rep / "rag.md", "Automatic checks").to_dict("records"),
                "llama": table(rep / "rag_llama32_3b.md", "Automatic checks").to_dict("records")},
        "learning_curve": lc.to_dict("records"),
    })


@router.get("/dataset")
def dataset():
    return _dataset()


@lru_cache(maxsize=1)
def _dataset():
    cfg = STATE["cfg"]
    stats = json.loads((cfg["paths"]["reports"] / "build_stats.json").read_text())
    r = STATE["reports"]
    per_year = r.groupby(["file_year", "family"]).size().unstack(fill_value=0).reset_index()
    ata = r[r.split == "train"].groupby("family").ata2.value_counts(normalize=True).unstack(0).fillna(0)
    ata = ata.loc[ata.sum(1).sort_values(ascending=False).index[:15]].reset_index()
    ata["title"] = ata.ata2.map(title)
    cond = r.PartCondition.fillna("").replace("", "(blank)").value_counts().head(12).reset_index()
    refs = pd.Series([x for rs in r.refs for x in rs]).value_counts().head(12).reset_index()
    variants = r.groupby(["family", "variant"]).size().rename("reports").reset_index().sort_values("reports", ascending=False)
    return _plain({"counts": stats["counts"], "sanity_2025": stats["sanity_2025"],
                   "per_year": per_year.to_dict("records"), "ata_share": ata.to_dict("records"),
                   "conditions": cond.set_axis(["condition", "reports"], axis=1).to_dict("records"),
                   "references": refs.set_axis(["reference", "citations"], axis=1).to_dict("records"),
                   "variants": variants.head(15).to_dict("records"),
                   "split_sizes": r.split.value_counts().to_dict()})


@router.get("/feedback")
def feedback_list(include_simulated: bool = False, limit: int = 200):
    rows = store.all_rows()
    if not include_simulated:
        rows = rows[rows.source != "simulated"]
    counts = store.all_rows().source.value_counts().to_dict()
    return _plain({"rows": rows.tail(limit).iloc[::-1].replace({np.nan: None}).to_dict("records"), "counts": counts})


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    rec = req.model_dump()
    if rec["system_correct"] is not None:
        rec["system_correct"] = int(rec["system_correct"])
    return {"id": store.add(rec)}


app.include_router(router, prefix="/api")
app.include_router(router)  # backward-compatible un-prefixed routes

_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _DIST.exists():  # built console: one server for API + UI
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="console")
