"""Final results: python -m aerofix.eval.final [--full]

Default (~1 min): refresh the classification report/figures from saved predictions,
compute the test-set ablations that saved predictions allow (DistilBERT leakage and
cross-family transfer), and assemble reports/RESULTS.md from every phase's outputs.
--full: first re-run every phase end to end in order (~4-5 h on the reference
laptop; needs the raw data, a CUDA GPU and Ollama), then assemble.

All numbers are on the 2025 test set unless a table says otherwise. Nothing here
changes a model or a choice; selection always happened on val (2024).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from aerofix.classify.data import LabelSpace, load_frame
from aerofix.classify.metrics import summary
from aerofix.config import load_config
from aerofix.utils import get_logger, md_table

log = get_logger(__name__)

PIPELINE = [
    ["download_data.py"], ["-m", "aerofix.data.build"], ["-m", "aerofix.eval.eda"],
    ["-m", "aerofix.preprocess.build_norm"], ["-m", "aerofix.eval.normalisation"],
    ["-m", "aerofix.segment.llm_label"], ["-m", "aerofix.segment.evaluate"], ["-m", "aerofix.segment.apply"],
    ["-m", "aerofix.classify.prepare"], ["-m", "aerofix.classify.tfidf"], ["-m", "aerofix.classify.bilstm"],
    ["-m", "aerofix.classify.transformer", "--model", "distilbert"],
    ["-m", "aerofix.classify.transformer", "--model", "roberta"],
    ["-m", "aerofix.classify.transformer", "--model", "distilbert", "--text-col", "text", "--epochs", "1",
     "--name", "distilbert_full_unmasked"],
    ["-m", "aerofix.classify.transformer", "--model", "distilbert", "--text-col", "text_masked", "--epochs", "1",
     "--name", "distilbert_full_masked"],
    ["-m", "aerofix.classify.transformer", "--model", "distilbert", "--train-family", "A320", "--epochs", "2",
     "--name", "distilbert_A320"],
    ["-m", "aerofix.classify.transformer", "--model", "distilbert", "--train-family", "B737", "--epochs", "2",
     "--name", "distilbert_B737"],
    ["-m", "aerofix.retrieve.evaluate"], ["-m", "aerofix.recommend.evaluate"], ["-m", "aerofix.rag.docs"],
    ["-m", "aerofix.rag.evaluate"], ["-m", "aerofix.rag.evaluate", "--model", "llama3.2:3b"],
    ["-m", "aerofix.feedback.simulate"],
]

LIMITATIONS = """- **No OEM manuals.** Airbus/Boeing AMM, SRM, FIM and TSM content is proprietary and was not used. AeroFix only points to references technicians cited in public reports; RAG passages come from generic FAA handbooks, which rarely match a specific aircraft system (e.g. door dampers).
- **US operators only.** FAA SDRs are filed by US operators and repair stations, so fleets, reporting habits and vocabulary are US-specific; transfer to other regions is untested.
- **Proxy labels.** Retrieval relevance (same JASC + condition) and action/reference hits are proxies for usefulness. A 50-query LLM check agreed with the retrieval proxy on only 63% of pairs; no large human-judged set exists.
- **Silver segmentation labels.** 399 of 400 problem/action annotations were made by a local LLM; segmentation scores measure agreement with those labels, not human gold.
- **Label noise in the source.** The same location can be coded to different ATA chapters by different reporters (see error analysis), which caps achievable accuracy.
- **Drift.** Part-condition codes drift across years (OTHER: 15% of train vs 24% of 2025); the feedback loop shows periodic updates are needed.
- **LLM latency and judgement.** The full Qwen 7B summary takes ~5-6 s (streamed); automatic checks catch invented citations/references but not subtle misreadings, so faithfulness still needs human scoring.
- **Decision support only.** AeroFix must never replace approved maintenance data or engineering judgement."""


def section(path, heading: str) -> str:
    """Markdown from the '## heading...' line up to the next '## ' heading (exclusive)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## ") and heading in ln), None)
    if start is None:
        return f"_(section '{heading}' not found in {path.name})_\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start + 1:end]).strip()
    return body.replace("](figures/", "](figures/") + "\n"


def table(path, heading: str) -> pd.DataFrame:
    """First markdown table inside a report section, as a DataFrame (numbers converted where possible)."""
    rows = [ln for ln in section(path, heading).splitlines() if ln.startswith("|")]
    if len(rows) < 3:
        return pd.DataFrame()
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]
    df = pd.DataFrame(cells[2:], columns=cells[0])
    for col in df.columns:
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().all():
            df[col] = conv
    return df


def headline(rep, cfg, cls_tab) -> str:
    """Baseline vs best per component, read from the phase reports (so a re-run updates it)."""
    seg = table(rep / "segmentation.md", "Results")
    seg = seg[seg.split == "test"].set_index("method").span_micro_F1
    seg_best = json.loads((cfg["paths"]["models"] / "segment" / "best.json").read_text())["best"]
    ret = table(rep / "retrieval.md", "Test (").set_index("method")["NDCG@5"]
    app_ret = json.loads((cfg["paths"]["models"] / "retrieve" / "best.json").read_text())["best"]
    act = table(rep / "recommend.md", "Corrective-action").set_index("method")["hit@3"]
    ref = table(rep / "recommend.md", "Manual-reference").set_index("method")["hit@3"]
    rag = table(rep / "rag.md", "Automatic checks").set_index("metric")["value"]
    seg_name = next(k for k in seg.index if k.startswith(seg_best))
    d = cls_tab.loc["DistilBERT (multi-task)"]
    act_best = act.drop(index=[k for k in act.index if k.startswith(("Global", "Most frequent"))]).idxmax()
    rows = [
        ("ATA chapter classification", f"majority {cls_tab.loc['Majority class', 'ata2 macro-F1']:.3f}; TF-IDF + LR "
         f"{cls_tab.loc['TF-IDF + LR', 'ata2 macro-F1']:.3f}", f"DistilBERT {d['ata2 macro-F1']:.3f} "
         f"(top-3 accuracy {d['ata2 top-3 acc']:.3f})", "macro-F1"),
        ("Part condition", f"majority {cls_tab.loc['Majority class', 'condition macro-F1']:.3f}",
         f"DistilBERT {d['condition macro-F1']:.3f}", "macro-F1"),
        ("Problem/action segmentation", f"rules {seg[[k for k in seg.index if k.startswith('rules')][0]]:.3f}",
         f"{seg_name} {seg[seg_name]:.3f}", "span F1 (silver labels)"),
        ("Similar-case retrieval", f"BM25 raw {ret['BM25 raw (tokenised)']:.3f}",
         f"{ret.idxmax()} {ret.max():.3f}; app ({app_ret}) {ret[app_ret]:.3f}", "NDCG@5 (proxy relevance)"),
        ("Corrective-action recommendation", f"global most frequent {act['Global most frequent actions']:.3f}",
         f"{act_best} {act[act_best]:.3f}", "hit@3"),
        ("Manual-reference recommendation", f"global most cited {ref['Global most cited']:.3f}",
         f"retrieval + count {ref['Retrieval + count']:.3f}; + specificity {ref['Retrieval + specificity (log-lift)']:.3f}",
         "hit@3"),
        ("Cited LLM summary (50 faults)", "-", f"Qwen 2.5 7B: {rag['summaries with >= 1 citation']} cite, "
         f"{rag['citation validity']} valid, {rag['summaries with a hallucinated reference']} invented refs",
         "automatic checks"),
    ]
    return md_table(pd.DataFrame(rows, columns=["Component", "Baseline", "Best", "Metric"]), index=False)


def run_full(cfg) -> None:
    for step in PIPELINE:
        cmd = [sys.executable, *step]
        log.info("running: %s", " ".join(step))
        if subprocess.run(cmd, cwd=cfg["project_root"]).returncode != 0:
            raise SystemExit(f"step failed: {' '.join(step)}")


def test_ablations(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    cdir = cfg["paths"]["models"] / "classify"
    labels = LabelSpace.load(cdir / "labels.json")
    df = load_frame(cfg).set_index("row_id")
    leak = []
    for run, desc in (("distilbert_full_unmasked", "full report, references UNMASKED (1 epoch)"),
                      ("distilbert_full_masked", "full report, references masked (1 epoch)"),
                      ("distilbert", "problem text only, masked (deployed model, 4 epochs)")):
        p = cdir / run / "preds_test.npz"
        if p.exists():
            z = np.load(p)
            ev = df.loc[z["row_id"]]
            leak.append({"DistilBERT input": desc, **summary(labels.encode("ata2", ev.ata2), z["ata2"].astype(np.float32))})
    fam = []
    for tr in ("A320", "B737"):
        p = cdir / f"distilbert_{tr}" / "preds_test.npz"
        if p.exists():
            z = np.load(p)
            ev = df.loc[z["row_id"]]
            for te in ("A320", "B737"):
                m = (ev.family == te).values
                s = summary(labels.encode("ata2", ev[m].ata2), z["ata2"].astype(np.float32)[m])
                fam.append({"train on": tr, "test on": te, "macro_F1": s["macro_F1"], "accuracy": s["accuracy"]})
    return pd.DataFrame(leak), pd.DataFrame(fam)


def error_cases(cfg) -> pd.DataFrame:
    cdir = cfg["paths"]["models"] / "classify"
    labels = LabelSpace.load(cdir / "labels.json")
    ann = pd.read_csv(cfg["paths"]["annotations"] / "error_analysis.csv")
    z = np.load(cdir / "distilbert" / "preds_test.npz")
    probs = pd.DataFrame({"row_id": z["row_id"], "pred_idx": z["ata2"].astype(np.float32).argmax(1),
                          "conf": z["ata2"].astype(np.float32).max(1)})
    df = load_frame(cfg)[["row_id", "ata2", "problem_masked"]]
    out = ann.merge(probs, on="row_id").merge(df, on="row_id")
    names = labels.names("ata2")
    out["true"] = [names[i] for i in labels.encode("ata2", out.ata2)]
    out["predicted"] = [names[i] for i in out.pred_idx]
    out["problem text"] = out.problem_masked.str.slice(0, 90)
    out["conf"] = out.conf.map(lambda c: f"{c:.2f}")
    return out[["problem text", "true", "predicted", "conf", "category", "explanation"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-run every phase end to end first (~4-5 h)")
    args = ap.parse_args()
    cfg = load_config()
    rep = cfg["paths"]["reports"]
    if args.full:
        run_full(cfg)
    subprocess.run([sys.executable, "-m", "aerofix.classify.report"], cwd=cfg["project_root"], check=True)

    stats = json.loads((rep / "build_stats.json").read_text())
    c = stats["counts"]
    res = pd.read_csv(rep / "classification_results.csv")
    test = res[res.split == "test"]
    cls_tab = test.pivot_table(index="model", columns="target", values="macro_F1").round(4)
    cls_tab = cls_tab[["ata2", "ata4", "condition"]].sort_values("ata2", ascending=False)
    acc = test[test.target == "ata2"].set_index("model")[["accuracy", "top3_acc"]].rename(
        columns={"accuracy": "ata2 accuracy", "top3_acc": "ata2 top-3 acc"})
    cls_tab = cls_tab.join(acc).rename(columns={"ata2": "ata2 macro-F1", "ata4": "ata4 macro-F1",
                                                "condition": "condition macro-F1"})
    leak, fam = test_ablations(cfg)
    errs = error_cases(cfg)
    cat_counts = errs.category.value_counts().rename("cases").rename_axis("error category").reset_index()
    rag_q = section(rep / "rag.md", "Automatic checks")
    rag_l = section(rep / "rag_llama32_3b.md", "Automatic checks")

    md = [
        "# AeroFix: final results\n\n",
        "Generated by `python -m aerofix.eval.final` (add `--full` to re-run every phase from raw data first). "
        "All numbers are on the **2025 test year** unless stated; every model choice was made on 2024 (val). "
        "Per-phase details: `reports/eda.md`, `normalisation.md`, `segmentation.md`, `classification.md`, "
        "`retrieval.md`, `recommend.md`, `rag.md`, `feedback_loop.md`.\n\n",
        "## 1. Data\n\n",
        f"FAA Service Difficulty Reports 1995-2025, filtered to the A320 family and Boeing 737: "
        f"{c['filtered_rows']:,} reports -> {c['final_rows']:,} after removing {c['dropped_duplicates']:,} duplicates "
        f"({c['dropped_cross_year_duplicates']:,} were later-year copies that would have leaked across splits). "
        f"Time-based split: train <= 2023 ({c['rows_per_split']['train']:,}), val 2024 ({c['rows_per_split']['val']:,}), "
        f"test 2025 ({c['rows_per_split']['test']:,}). The raw 2025 counts match the project brief exactly "
        "(67,652 rows; 22,935 B737; 12,415 A320; 38 chapters; 347 JASC codes).\n\n",
        "## 2. Headline results (2025 test)\n\n", headline(rep, cfg, cls_tab), "\n\n",
        "## 3. Classification (2025 test, frozen models)\n\n", md_table(cls_tab), "\n\n",
        "![confusion ata2](figures/confusion_ata2.png)\n\n",
        "### 3a. Leakage ablation (DistilBERT, 2025 test)\n\nManual references such as `AMM 52-10-00` contain the "
        "ATA chapter. Masking them removes the shortcut; restricting the input to the problem text (what a technician "
        "knows before fixing) removes the action text as well.\n\n", md_table(leak, index=False), "\n\n",
        "### 3b. Cross-family transfer (DistilBERT, equal train size, 2025 test)\n\n", md_table(fam, index=False), "\n\n",
        "### 3c. Input and normalisation ablations (TF-IDF + LR, ata2, 150k train sample, **val 2024**)\n\n",
        section(rep / "classification.md", "4. Ablations"), "\n",
        "## 4. Text normalisation (Phase 2, val)\n\n",
        "### Language model\n\n", section(rep / "normalisation.md", "2. N-gram language model"), "\n",
        "### Spelling correction\n\n", section(rep / "normalisation.md", "3. Spelling correction").split("**Real val text**")[0], "\n",
        "## 5. Problem / action segmentation (Phase 3)\n\n", section(rep / "segmentation.md", "Results"), "\n",
        section(rep / "segmentation.md", "Annotation"), "\n",
        "## 6. Retrieval (2025 test)\n\n", section(rep / "retrieval.md", "Test ("), "\n",
        section(rep / "retrieval.md", "LLM sanity check"), "\n",
        "## 7. Recommenders (2025 test)\n\n",
        section(rep / "recommend.md", "Corrective-action recommendation"), "\n",
        section(rep / "recommend.md", "Manual-reference recommendation"), "\n",
        "## 8. Cited LLM summaries (50 test faults)\n\n**Qwen 2.5 7B (app default)**\n\n", rag_q,
        "\n**Llama 3.2 3B**\n\n", rag_l, "\n",
        "## 9. Feedback learning loop\n\n![learning curve](learning_curve.png)\n\n",
        section(rep / "feedback_loop.md", "After 3 rounds"), "\n",
        "## 10. Error analysis: 20 random DistilBERT test mistakes\n\n",
        f"DistilBERT gets the ATA chapter wrong for "
        f"{1 - cls_tab.loc['DistilBERT (multi-task)', 'ata2 accuracy']:.1%} of 2025 test reports. Twenty random "
        "mistakes were read and explained by hand "
        "(`data/annotations/error_analysis.csv`):\n\n", md_table(cat_counts, index=False), "\n\n",
        md_table(errs, index=False), "\n\n",
        "**Takeaways.** Most mistakes are not model failures in the usual sense: structure at system boundaries "
        "(center wing box, wheel wells, door surrounds) is coded inconsistently by reporters, some reports use "
        "operator-specific conventions, and a few labels look wrong. Genuine model errors are keyword bias (DOOR, "
        "WING pulling predictions) and missing domain knowledge (floor-path lighting in a 'transition strip'). One "
        "error comes from Phase 3 segmentation cutting the problem text at 'REPAIR'.\n\n",
        "## 11. Limitations\n\n", LIMITATIONS, "\n",
    ]
    out = rep / "RESULTS.md"
    out.write_text("".join(md), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
