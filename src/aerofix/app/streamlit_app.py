"""AeroFix UI: streamlit run src/aerofix/app/streamlit_app.py

Loads the Advisor once (st.cache_resource), shows the structured results as soon as
they are ready (~0.15 s) and then streams the LLM summary. Results live in session
state, so interacting with the feedback form doesn't re-run the models.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from aerofix.config import load_config
from aerofix.feedback import store
from aerofix.pipeline import DISCLAIMER, Advisor
from aerofix.rag.ata import ATA_TITLES
from aerofix.rag.generate import generate_stream, trim_to_sentence, validate

STAGES = {"IN": "Inspection / maintenance", "TX": "Taxi", "TO": "Takeoff", "CL": "Climb", "CR": "Cruise",
          "DE": "Descent", "AP": "Approach", "LD": "Landing"}
EXAMPLES = ["L1 door damper leaking, emer cyl press low", "FWD lav smoke detector inop",
            "Corrosion found on frame 16 lower flange at floor beam", "Emergency exit light battery pack failed capacity test"]
STOP = {"THE", "AND", "WITH", "FOR", "FROM", "WAS", "ARE", "NOT", "REF", "ON", "AT", "IN", "OF", "TO"}

st.set_page_config(page_title="AeroFix", page_icon=":material/build:", layout="wide")


@st.cache_resource(show_spinner="Loading AeroFix models (first start takes about 30 s)...")
def get_advisor() -> Advisor:
    return Advisor()


@st.cache_data
def variants() -> dict[str, list[str]]:
    df = pd.read_parquet(load_config()["paths"]["processed"] / "sdr_a320_b737.parquet", columns=["family", "variant"])
    return {f: g.variant.value_counts().index.tolist() for f, g in df.groupby("family")}


def highlight(text: str, query_words: set[str]) -> str:
    """Bold the words a past case shares with the query (why it was retrieved)."""
    return re.sub(r"[A-Z0-9][A-Z0-9/&-]*", lambda m: f"**{m.group(0)}**" if m.group(0) in query_words else m.group(0),
                  text.replace("*", ""))


# ---------------- input ----------------
st.title("AeroFix", anchor=False)
st.caption("Maintenance fault advisor for A320-family and Boeing 737 fleets, built on public FAA Service Difficulty Reports.")
st.warning(DISCLAIMER, icon=":material/warning:", title="Decision support only")

with st.form("query"):
    c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    family = c1.segmented_control("Aircraft family", ["A320", "B737"], default="A320", required=True)
    variant = c2.selectbox("Variant", variants()[family])
    stage = c3.selectbox("Stage of operation (optional)", [None, *STAGES], format_func=lambda s: "Not specified" if s is None else f"{s} - {STAGES[s]}",
                         help="Recorded with your feedback; not used by the models.")
    text = st.text_area("Fault description", value=EXAMPLES[0], height=90, max_chars=2000,
                        help="Free text as written in the tech log; abbreviations and typos are fine.")
    submitted = st.form_submit_button("Get advice", type="primary", icon=":material/search:")
st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLES[1:]))

advisor = get_advisor()
if submitted and text.strip():
    st.session_state.result = advisor.advise(family, variant, text, stage=stage, llm=False)
    st.session_state.summary = None  # stream a new one below
    st.session_state.feedback_saved = False

res = st.session_state.get("result")
if res is None:
    st.info("Describe a fault and press **Get advice**.", icon=":material/info:")
    st.stop()

advice_tab, hood_tab = st.tabs([":material/build: Advice", ":material/science: Under the hood"])

with advice_tab:
    # ---- predictions ----
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True, height="stretch"):
        ata, p, ttl = res["system"]["ata2"][0]
        st.markdown(f"**Predicted system**  \nATA {ata} · {ttl}")
        st.progress(p, text=f"{p:.0%} confidence")
        st.caption("Next: " + ", ".join(f"ATA {a} {t.title()} ({q:.0%})" for a, q, t in res["system"]["ata2"][1:]))
        st.caption(f"JASC subsystem: {res['system']['ata4'][0][0]} ({res['system']['ata4'][0][1]:.0%})")
    with c2.container(border=True, height="stretch"):
        cond, p = res["condition"][0]
        st.markdown(f"**Predicted part condition**  \n{cond.title()}")
        st.progress(p, text=f"{p:.0%} confidence")
        st.caption("Next: " + ", ".join(f"{c.title()} ({q:.0%})" for c, q in res["condition"][1:]))
    with c3.container(border=True, height="stretch"):
        st.markdown("**Most common corrective actions** in similar cases")
        if res["actions"]:
            for a, share in res["actions"][:4]:
                st.progress(min(share, 1.0), text=f"{a.title()} · {share:.0%}")
        else:
            st.caption("No corrective actions recorded in the similar cases.")

    # ---- references ----
    with st.container(border=True):
        st.markdown("**Manual references technicians cited** for this kind of fault (ranked by specificity)")
        rows = [{"Reference": r["reference"], "Cited in similar cases": r["cited_in_share"],
                 "Cited across all reports": r["global_share"],
                 "Type": "generic" if r["global_share"] > 0.02 else "specific"} for r in res["references"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Recommended manual references",
                         column_config={"Cited in similar cases": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                        "Cited across all reports": st.column_config.NumberColumn(format="percent")})
            st.caption("References are pointers only: the procedure itself must come from the approved OEM data.")
        else:
            st.caption("No references cited in the similar cases.")

    # ---- LLM summary (streamed once per query) ----
    with st.container(border=True):
        st.markdown("**Summary** · local LLM, cited")
        if st.session_state.get("summary") is None:
            slot = st.empty()
            with slot:
                try:
                    raw = st.write_stream(generate_stream(res["evidence"]))
                except Exception as e:  # Ollama not running
                    raw = ""
                    st.error(f"LLM unavailable: {e}", icon=":material/error:")
            text_final = trim_to_sentence(raw) if raw else ""
            st.session_state.summary = {"text": text_final, **validate(text_final, res["evidence"])}
            slot.empty()
        s = st.session_state.summary
        if s["text"]:
            st.markdown(s["text"])
            if s["ok"]:
                st.badge(f"{s['n_citations']} citations, all found in the evidence", icon=":material/verified:", color="green")
            else:
                problems = ([f"invalid citation(s): {', '.join(s['invalid_citations'])}"] if s["invalid_citations"] else []) + \
                           ([f"reference(s) not in evidence: {', '.join(s['hallucinated_refs'])}"] if s["hallucinated_refs"] else []) + \
                           (["no citations"] if not s["has_citation"] else [])
                st.badge("Check: " + "; ".join(problems), icon=":material/report:", color="orange")

    # ---- evidence ----
    st.subheader("Similar past reports", anchor=False)
    qwords = {w for w in re.findall(r"[A-Z0-9][A-Z0-9/&-]*", res["under_the_hood"]["normalised"]) if len(w) > 2 and w not in STOP}
    for i, c in enumerate(res["cases"], 1):
        with st.expander(f"Case {i} · {c['variant']} · {c['year']} · similarity {c['similarity']:.2f}"):
            st.markdown("**Problem:** " + highlight(c["problem"], qwords))
            st.markdown("**Action:** " + (c["action"] or "_not recorded_"))
            if c["refs"]:
                st.caption("References cited: " + ", ".join(c["refs"]))
            st.caption(f"Canonical actions: {', '.join(c['actions']) or '-'} · SDR row {c['row_id']}")

    st.subheader("Related FAA guidance (public)", anchor=False)
    for p in res["passages"]:
        with st.expander(f"{p['label']} · {p['heading']}"):
            st.write(p["text"])

    # ---- feedback ----
    st.subheader("Feedback", anchor=False)
    if st.session_state.get("feedback_saved"):
        st.success("Thanks, feedback saved. It is used to retrain the models (Phase 9).", icon=":material/check:")
    else:
        with st.form("feedback"):
            ok = st.segmented_control("Was the predicted system correct?", ["Correct", "Wrong"])
            chapters = [f"{k} {v}" for k, v in ATA_TITLES.items()]
            correct = st.selectbox("If wrong, the correct ATA chapter", [None, *chapters],
                                   format_func=lambda x: "-" if x is None else x)
            actions = [a for a, _ in res["actions"]] + ["Something else"]
            fixed = st.selectbox("Which action fixed it?", [None, *actions], format_func=lambda x: "-" if x is None else x.title())
            comment = st.text_input("Comment (optional)")
            if st.form_submit_button("Send feedback", icon=":material/send:"):
                pred = res["system"]["ata2"][0][0]
                store.add({"family": res["input"]["family"], "variant": res["input"]["variant"],
                           "stage": res["input"]["stage"], "text": res["input"]["text"], "predicted_ata2": pred,
                           "predicted_condition": res["condition"][0][0],
                           "system_correct": None if ok is None else int(ok == "Correct"),
                           "correct_ata2": pred if ok == "Correct" else (correct.split()[0] if correct else None),
                           "action_fixed": fixed, "comment": comment or None})
                st.session_state.feedback_saved = True
                st.rerun()

with hood_tab:
    h = res["under_the_hood"]
    st.markdown("**Text processing**")
    st.dataframe(pd.DataFrame({"step": ["input", "cleaned (boilerplate stripped)", "references masked",
                                        "abbreviations + spelling", "lemmatised (BM25 input)"],
                               "text": [res["input"]["text"], h["cleaned"], h["masked"], h["normalised"], h["lemmatised"]]}),
                 hide_index=True, alt="Text normalisation steps")
    if h["context_prefix"] != "NONE":
        st.caption(f"Boilerplate prefix detected: {h['context_prefix']}")
    st.markdown("**Model scores (top 3 per head, multi-task DistilBERT)**")
    c1, c2, c3 = st.columns(3)
    c1.dataframe(pd.DataFrame(res["system"]["ata2"], columns=["ATA", "p", "title"]), hide_index=True, alt="ATA chapter scores")
    c2.dataframe(pd.DataFrame(res["system"]["ata4"], columns=["JASC", "p"]), hide_index=True, alt="JASC scores")
    c3.dataframe(pd.DataFrame(res["condition"], columns=["condition", "p"]), hide_index=True, alt="Condition scores")
    st.markdown("**Timings (ms)**")
    st.dataframe(pd.DataFrame([res["timings"]]), hide_index=True, alt="Pipeline timings")
    st.markdown("**Exact evidence sent to the LLM**")
    st.code(res["evidence"].render(), language=None, wrap_lines=True)
