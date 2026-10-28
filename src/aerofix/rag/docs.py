"""Parse public FAA documents into cited passages: python -m aerofix.rag.docs

1. pymupdf reads every line with its font size and weight.
2. A line is a section heading if it is short, all of its text is bold, and its
   font is at least the document's body size (the most common size).
3. Text between headings forms a section; sections are cut into ~300-word chunks
   with 50 words of overlap (so a sentence split at a boundary still appears whole
   in one chunk). Each chunk keeps: document short name, page of its first line,
   and the section heading.
4. Chunks are embedded with all-MiniLM-L6-v2 (zero-shot: the fine-tuned model was
   trained on SDR fault pairs, not handbook prose) and stored in a FAISS index.
Scanned PDFs without a text layer (the TCDS) produce no text and are skipped.
Writes models/rag/passages.parquet and models/rag/passages.faiss.
"""
from __future__ import annotations

import re
from collections import Counter

import faiss
import numpy as np
import pandas as pd
import pymupdf

from aerofix.config import load_config
from aerofix.utils import get_logger

log = get_logger(__name__)
SHORT_NAMES = {"FAA-H-8083-30B_AMT_General.pdf": "AMT General", "FAA-H-8083-31B_AMT_Airframe.pdf": "AMT Airframe",
               "FAA-H-8083-32B_AMT_Powerplant.pdf": "AMT Powerplant", "AC_43.13-1B_w-chg1.pdf": "AC 43.13-1B"}
WORDS, OVERLAP = 300, 50


def lines(doc: pymupdf.Document):
    """Yield (page_number, text, is_heading) for every text line."""
    sizes = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    sizes[round(s["size"])] += len(s["text"].strip())
    body = sizes.most_common(1)[0][0] if sizes else 10
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                spans = [s for s in ln["spans"] if s["text"].strip()]
                text = " ".join(s["text"].strip() for s in spans).strip()
                # soft hyphens / undecodable glyphs from line-end hyphenation ("dis\xad charge")
                text = re.sub(r"[­�]\s*", "", text)
                if not text:
                    continue
                heading = (3 <= len(text) <= 80 and all(s["flags"] & 16 for s in spans)
                           and min(round(s["size"]) for s in spans) >= body and re.search(r"[A-Za-z]{3}", text))
                yield pno, text, bool(heading)


def chunk_document(path) -> list[dict]:
    doc = pymupdf.open(path)
    name = SHORT_NAMES.get(path.name, path.stem)
    out, heading = [], ""
    words: list[tuple[str, int]] = []  # (word, page)

    def flush():
        i = 0
        while i < len(words):
            piece = words[i:i + WORDS]
            if len(piece) >= 40:  # skip tiny fragments (headers, captions)
                out.append({"doc": name, "page": piece[0][1], "heading": heading,
                            "text": " ".join(w for w, _ in piece)})
            if i + WORDS >= len(words):
                break
            i += WORDS - OVERLAP

    for pno, text, is_heading in lines(doc):
        if is_heading:
            flush()
            words, heading = [], text
        else:
            words += [(w, pno) for w in text.split()]
    flush()
    return out


def main() -> None:
    cfg = load_config()
    from sentence_transformers import SentenceTransformer
    from aerofix.retrieve.train import encode
    rows = []
    for path in sorted(cfg["paths"]["raw_docs"].glob("*.pdf")):
        ch = chunk_document(path)
        log.info("%s: %d chunks", path.name, len(ch))
        rows += ch
    df = pd.DataFrame(rows)
    df["label"] = df.doc + " p." + df.page.astype(str)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = encode(enc, list((df.heading + ". " + df.text).str.slice(0, 2000)))
    out = cfg["paths"]["models"] / "rag"
    out.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(vec.astype(np.float32))
    faiss.write_index(index, str(out / "passages.faiss"))
    df.to_parquet(out / "passages.parquet", index=False)
    log.info("indexed %d passages from %d documents", len(df), df.doc.nunique())


if __name__ == "__main__":
    main()
