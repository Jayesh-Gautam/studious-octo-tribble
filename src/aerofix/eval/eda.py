"""Exploratory data analysis: python -m aerofix.eval.eda

Reads data/processed/sdr_a320_b737.parquet and writes reports/eda.md plus figures
in reports/figures/. Statistics that drive modelling choices (top-K chapters,
abbreviation candidates) are computed on TRAIN only, so the test year never
influences design decisions.
"""
from __future__ import annotations

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import pandas as pd

from aerofix.config import load_config
from aerofix.preprocess.refs import ref_chapters
from aerofix.utils import get_logger, md_table

log = get_logger(__name__)

# Validated categorical palette (slot 1 blue, slot 2 orange), fixed order: B737 then A320.
COLORS = {"B737": "#2a78d6", "A320": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACTION_VERBS = re.compile(
    r"\b(?:REPLACED|REMOVED|INSTALLED|REPAIRED|SERVICED|LUBRICATED|RESEATED|ADJUSTED|"
    r"TIGHTENED|CLEANED|RESET|SECURED|REWORKED|BLENDED|DEFERRED|TROUBLESHOT|R&R|RPLCD|REPLCD)\b"
)
TOKEN = re.compile(r"[A-Z0-9&/]+")
VOWELS = set("AEIOU")


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive axes: light grid behind the marks, no top/right spines, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path) -> str:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"![{path.stem}](figures/{path.name})"


def plot_rows_per_year(df, fig_dir) -> str:
    t = df.groupby(["file_year", "family"]).size().unstack(fill_value=0)[["B737", "A320"]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = None
    for fam in t.columns:
        ax.bar(t.index, t[fam], bottom=bottom, color=COLORS[fam], label=fam, width=0.8,
               edgecolor=SURFACE, linewidth=1)
        bottom = t[fam] if bottom is None else bottom + t[fam]
    for boundary in (2023.5, 2024.5):  # train | val | test
        ax.axvline(boundary, color=INK_2, linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Reports per year after cleaning and dedup (dashed: split boundaries)", "", "reports")
    return _save(fig, fig_dir / "rows_per_year.png")


def plot_ata_by_family(train, fig_dir, top: int = 20) -> str:
    share = train.groupby("family").ata2.value_counts(normalize=True).unstack(0).fillna(0)[["B737", "A320"]]
    share = share.loc[share.sum(axis=1).sort_values(ascending=False).index[:top]][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    h = 0.4
    for i, fam in enumerate(share.columns):
        ax.barh([j + (i - 0.5) * h for j in range(len(share))], share[fam] * 100, height=h * 0.9,
                color=COLORS[fam], label=fam)
    ax.set_yticks(range(len(share)), [f"ATA {c}" for c in share.index])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, f"Top {top} ATA chapters, share of each family's train reports", "% of family's reports")
    return _save(fig, fig_dir / "ata_by_family.png")


def plot_hbar(series, title, xlabel, path, color=COLORS["B737"]) -> str:
    s = series[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1))
    ax.barh(range(len(s)), s.values, color=color, height=0.7)
    ax.set_yticks(range(len(s)), s.index)
    _style(ax, title, xlabel)
    return _save(fig, path)


def plot_text_length(df, fig_dir) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = range(0, 151, 3)
    for split, color in (("train", COLORS["B737"]), ("test", COLORS["A320"])):
        n = df.n_words[df.split == split].clip(upper=150)
        ax.hist(n, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                label=f"{split} (median {int(df.n_words[df.split == split].median())} words)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Report length in words (clipped at 150)", "words", "density")
    return _save(fig, fig_dir / "text_length.png")


def plot_ref_rate(df, fig_dir) -> str:
    r = df.assign(has=df.refs.str.len() > 0).groupby(["file_year", "family"]).has.mean().unstack()[["B737", "A320"]]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for fam in r.columns:
        ax.plot(r.index, r[fam] * 100, color=COLORS[fam], linewidth=2, marker="o", markersize=3, label=fam)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Share of reports citing at least one manual reference", "", "%")
    return _save(fig, fig_dir / "ref_rate_by_year.png")


def leakage_stats(df) -> pd.DataFrame:
    """How often the cited reference chapter equals the ATA label: the leak we mask."""
    rows = []
    for split in ("train", "val", "test"):
        d = df[df.split == split]
        has = d.refs.str.len() > 0
        sub = d[has]
        any_match = [r in ref_chapters(refs) for r, refs in zip(sub.ata2, sub.refs)]
        first_match = [ref_chapters(refs)[0] == r for r, refs in zip(sub.ata2, sub.refs)]
        leftover = d.text_masked.str.extract(r"\b(\d{2})-\d{2}\b", expand=False)
        rows.append({
            "split": split, "rows": len(d), "cite_ref_%": round(100 * has.mean(), 1),
            "any_ref_chapter_=_ata2_%": round(100 * pd.Series(any_match).mean(), 1),
            "first_ref_chapter_=_ata2_%": round(100 * pd.Series(first_match).mean(), 1),
            "residual_leak_after_mask_%": round(100 * (leftover == d.ata2).mean(), 2),
        })
    return pd.DataFrame(rows)


def abbreviation_candidates(train, n: int = 40) -> pd.DataFrame:
    """Frequent short tokens with no vowels or with & or / (RPLCD, R&R, W/, CHK):
    the starting list for the Phase 2 abbreviation dictionary."""
    c = Counter()
    for t in train.text_masked:
        c.update(TOKEN.findall(t))
    cand = [(tok, k) for tok, k in c.most_common(5000)
            if 2 <= len(tok) <= 8 and not tok.isdigit() and not any(ch.isdigit() for ch in tok)
            and ("&" in tok or "/" in tok or not (set(tok) & VOWELS))]
    return pd.DataFrame(cand[:n], columns=["token", "count"])


def main() -> None:
    cfg = load_config()
    reports = cfg["paths"]["reports"]
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg["paths"]["processed"] / "sdr_a320_b737.parquet")
    train = df[df.split == "train"]
    test25 = df[df.split == "test"]
    log.info("loaded %d rows", len(df))

    sections = ["# AeroFix EDA\n",
                "Generated by `python -m aerofix.eval.eda` from `data/processed/sdr_a320_b737.parquet` "
                "(1995-2025, A320 family + 737, after cleaning and deduplication). "
                "Design statistics (top-K chapters, abbreviation candidates) use **train only**.\n"]

    sections += ["## 1. Size per split and family\n",
                 md_table(df.groupby(["split", "family"]).size().unstack().reindex(["train", "val", "test"])), "\n",
                 plot_rows_per_year(df, fig_dir), "\n"]

    var = df.groupby(["family", "variant"]).size().rename("reports").reset_index().sort_values("reports", ascending=False)
    gen = df.groupby(["family", "generation"]).size().rename("reports").reset_index()
    sections += ["## 2. Variants\n", md_table(var.head(25), index=False), "\n\nBy generation:\n\n",
                 md_table(gen, index=False), "\n"]

    ata = train.ata2.value_counts()
    top_k = cfg["classify"]["top_k_ata2"]
    ata_tab = pd.DataFrame({"train_reports": ata, "share_%": (100 * ata / ata.sum()).round(1),
                            "cumulative_%": (100 * ata.cumsum() / ata.sum()).round(1)}).rename_axis("ata2")
    sections += [f"## 3. ATA chapters\n\nTrain has {ata.size} chapters and {train.ata4.nunique()} JASC codes; "
                 f"{int((ata >= 200).sum())} chapters have >= 200 train reports. The top {top_k} cover "
                 f"{ata_tab['cumulative_%'].iloc[top_k - 1]}% of train; the rest become `OTHER` in Phase 4.\n\n",
                 md_table(ata_tab.head(25)), "\n\n", plot_ata_by_family(train, fig_dir), "\n\n",
                 "2025 (test) top chapters, for comparison with the brief:\n\n",
                 md_table(test25.ata2.value_counts().head(8).rename("test_reports").rename_axis("ata2")), "\n"]

    sections += ["## 4. Text length\n",
                 md_table(df.groupby("split").n_words.describe().round(1).reindex(["train", "val", "test"])), "\n\n",
                 plot_text_length(df, fig_dir), "\n"]

    sections += ["## 5. Part conditions (train)\n",
                 plot_hbar(train.PartCondition.replace("", "(blank)").value_counts().head(20),
                           "Top 20 part conditions (train)", "reports", fig_dir / "part_conditions.png"), "\n"]

    refs = pd.Series([r for rs in train.refs for r in rs])
    sections += ["## 6. Manual references\n",
                 "Manual types cited (train):\n\n",
                 md_table(refs.str.split().str[0].value_counts().rename("citations").rename_axis("type")), "\n\n",
                 plot_hbar(refs.value_counts().head(20), "Top 20 cited references (train): generic SRM chapters dominate",
                           "citations", fig_dir / "top_refs.png"), "\n\n",
                 plot_ref_rate(df, fig_dir), "\n\n",
                 "### Label leakage\n\nIf the reference chapter equals the ATA label, a model can read the answer "
                 "off the text. Masking replaces references with `[REF]`. The last column is the share of rows "
                 "where a short `NN-NN` number matching the label still survives masking (unusual formats such as "
                 "`SRI 55-26`, `CDL 32-10`, `A321AMM 72-15`).\n\n",
                 md_table(leakage_stats(df), index=False), "\n"]

    has_action = df.text.str.contains(ACTION_VERBS)
    sections += ["## 7. Context prefixes and corrective actions\n",
                 md_table(df.context.value_counts().rename("reports").rename_axis("context")), "\n\n",
                 f"Share of reports containing a corrective-action verb (REPLACED, REMOVED, ...): "
                 f"train {100 * has_action[df.split == 'train'].mean():.1f}%, "
                 f"test {100 * has_action[df.split == 'test'].mean():.1f}%. "
                 "This is a rough check; Phase 3 segments problem vs action properly.\n"]

    sections += ["## 8. Abbreviation candidates (train)\n",
                 "Frequent short tokens with no vowels or with `&` / `/`: starting list for Phase 2.\n\n",
                 md_table(abbreviation_candidates(train), index=False), "\n"]

    out = reports / "eda.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    log.info("wrote %s and %d figures", out, len(list(fig_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
