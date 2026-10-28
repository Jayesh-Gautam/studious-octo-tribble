"""Annotation tool: streamlit run src/aerofix/segment/annotate_app.py

Each report is shown as chunks (split at punctuation, step markers, action verbs
and document words). Pick a label per chunk; if a label changes inside a chunk,
split it before the right word. Every save appends one JSON line to
data/annotations/segments.jsonl; the latest line per report wins, so you can go
back and redo one. Train/dev reports come pre-labelled by the rule baseline; test
reports start blank so the evaluation isn't biased towards the rules.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from aerofix.config import load_config
from aerofix.segment.common import LABELS, chunk_bounds, join_tokens
from aerofix.segment.rules import predict as rule_predict

COLORS = {"PROBLEM": "#eb6834", "ACTION": "#2a78d6", "REF": "#1baf7a", "OTHER": "#8a8984"}

cfg = load_config()
SAMPLE = cfg["paths"]["annotations"] / "seg_sample.parquet"
OUT = cfg["paths"]["annotations"] / "segments.jsonl"


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE)


def load_done() -> dict[int, dict]:
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["row_id"]] = rec
    return done


def init_report(row) -> None:
    """Fresh chunking + pre-fill for a report, stored in session state."""
    toks = list(row.tokens)
    st.session_state.bounds = chunk_bounds(toks)
    pre = rule_predict([toks])[0] if row.ann_split != "test" else [None] * len(toks)
    st.session_state.chunk_labels = [pre[b] for b in st.session_state.bounds]
    st.session_state.loaded_row = int(row.row_id)
    st.session_state.started = time.time()


def labels_from_chunks(n: int) -> list[str | None]:
    b = st.session_state.bounds + [n]
    out: list[str | None] = []
    for i, lab in enumerate(st.session_state.chunk_labels):
        out += [lab] * (b[i + 1] - b[i])
    return out


st.set_page_config(page_title="AeroFix annotation", layout="wide")
sample, done = load_sample(), load_done()
queue = [r for r in sample.itertuples() if r.row_id not in done]

st.title("Problem / action annotation")
st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} reports annotated")
with st.expander("Guide (docs/ANNOTATION_GUIDE.md)"):
    st.markdown((cfg["project_root"] / "docs" / "ANNOTATION_GUIDE.md").read_text(encoding="utf-8"))

if "redo" in st.session_state and st.session_state.redo is not None:
    current = sample[sample.row_id == st.session_state.redo].iloc[0]
elif queue:
    current = sample[sample.row_id == queue[0].row_id].iloc[0]
else:
    st.success("All reports annotated. Thank you!")
    st.stop()

if st.session_state.get("loaded_row") != int(current.row_id):
    init_report(current)

tokens = list(current.tokens)
st.caption(f"row_id {current.row_id} · {current.family} · {current.file_year} · "
           f"{'TEST: no pre-fill' if current.ann_split == 'test' else 'pre-filled by rules: check every chunk'}")

bounds = st.session_state.bounds
for ci, start in enumerate(bounds):
    end = bounds[ci + 1] if ci + 1 < len(bounds) else len(tokens)
    c1, c2, c3 = st.columns([6, 4, 2])
    words = " ".join(f"{tokens[i]}<sub style='color:#8a8984'>{i}</sub>" for i in range(start, end))
    c1.markdown(words, unsafe_allow_html=True)
    current_lab = st.session_state.chunk_labels[ci]
    choice = c2.radio("label", LABELS, index=LABELS.index(current_lab) if current_lab else None,
                      key=f"lab_{current.row_id}_{start}", horizontal=True, label_visibility="collapsed")
    st.session_state.chunk_labels[ci] = choice
    if end - start > 1:
        with c3.popover("split"):
            at = st.number_input("split before word #", min_value=start + 1, max_value=end - 1,
                                 value=start + 1, key=f"split_{current.row_id}_{start}")
            if st.button("split", key=f"do_split_{current.row_id}_{start}"):
                st.session_state.bounds.insert(ci + 1, int(at))
                st.session_state.chunk_labels.insert(ci + 1, choice)
                st.rerun()

labels = labels_from_chunks(len(tokens))
preview = " ".join(
    f"<span style='background:{COLORS[lab]}33;border-bottom:3px solid {COLORS[lab]}'>{t}</span>"
    if lab else f"<span style='border-bottom:3px dashed #c00'>{t}</span>"
    for t, lab in zip(tokens, labels))
st.markdown("**Preview** " + " ".join(f"<span style='color:{c}'>■ {k}</span>" for k, c in COLORS.items()),
            unsafe_allow_html=True)
st.markdown(preview, unsafe_allow_html=True)

b1, b2, b3, _ = st.columns([2, 2, 2, 6])
missing = sum(lab is None for lab in labels)
if b1.button("Save & next", type="primary", disabled=missing > 0,
             help="Every chunk needs a label" if missing else None):
    rec = {"row_id": int(current.row_id), "ann_split": current.ann_split, "tokens": tokens, "labels": labels,
           "prefilled": current.ann_split != "test", "skipped": False,
           "seconds": round(time.time() - st.session_state.started, 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    st.session_state.redo = None
    st.rerun()
if b2.button("Skip (unusable)"):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"row_id": int(current.row_id), "ann_split": current.ann_split, "tokens": tokens,
                            "labels": None, "skipped": True}) + "\n")
    st.rerun()
if done and b3.button("Redo last saved"):
    st.session_state.redo = list(done)[-1]
    st.session_state.loaded_row = None
    st.rerun()

st.caption(f"Plain text: {join_tokens(tokens)}")
