"""Thin HTTP API: uvicorn aerofix.app.api:app --port 8000

POST /advise    {"family": "A320", "variant": "A321", "text": "...", "stage": null, "llm": true}
                -> system, condition, actions, references, cases, passages, summary, timings
POST /feedback  {"text": "...", "family": ..., "system_correct": true, "action_fixed": "..."}
GET  /health
The Advisor (all models) is loaded once at startup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aerofix.feedback import store
from aerofix.pipeline import Advisor

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["advisor"] = Advisor()
    yield
    STATE.clear()


app = FastAPI(title="AeroFix", description="Maintenance fault advisor (decision support only)", lifespan=lifespan)


class AdviseRequest(BaseModel):
    family: str = Field(pattern="^(A320|B737)$")
    variant: str = ""
    text: str = Field(min_length=3, max_length=2000)
    stage: str | None = None
    llm: bool = True


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
    """numpy scalars / tuples -> JSON-friendly Python types."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "advisor" in STATE}


@app.post("/advise")
def advise(req: AdviseRequest):
    if "advisor" not in STATE:
        raise HTTPException(503, "models still loading")
    out = STATE["advisor"].advise(req.family, req.variant, req.text, stage=req.stage, llm=req.llm)
    out.pop("evidence")
    return _plain(out)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    rec = req.model_dump()
    if rec["system_correct"] is not None:
        rec["system_correct"] = int(rec["system_correct"])
    return {"id": store.add(rec)}
