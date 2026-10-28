# AeroFix architecture

Two halves: an **offline** pipeline that turns 31 years of FAA reports into models and indexes (each box is one `python -m ...` command), and an **online** request path that the Streamlit app and the FastAPI service share through one `Advisor` object (`src/aerofix/pipeline.py`).

## Offline: data → models

```mermaid
flowchart TD
    subgraph SRC[Public sources]
        SDR[(FAA SDR CSVs<br/>1995-2025)]
        DOCS[(FAA AMT Handbooks<br/>AC 43.13-1B PDFs)]
    end

    SDR --> BUILD["data.build<br/>filter A320/737, variant, ATA,<br/>clean, dedupe, time split"]
    BUILD --> REFS["preprocess.refs<br/>extract + mask manual references"]
    REFS --> NORM["preprocess.build_norm<br/>abbreviations, spelling (noisy channel<br/>+ Kneser-Ney trigram LM), stem / lemma"]
    REFS --> SEG["segment.*<br/>LLM silver labels → rules / CRF / DistilBERT<br/>→ problem_text, action_text"]
    NORM --> CLS
    SEG --> CLS["classify.*<br/>TF-IDF+LR · BiLSTM+Word2Vec ·<br/>DistilBERT / RoBERTa multi-task<br/>(ATA, JASC, condition)"]
    SEG --> RET["retrieve.evaluate<br/>BM25 · Word2Vec · MiniLM fine-tuned (MNRL)<br/>· cross-encoder · RRF hybrid"]
    NORM --> RET
    SEG --> REC["recommend.evaluate<br/>canonical actions (rules + clustering),<br/>reference global rates"]
    DOCS --> RAGD["rag.docs<br/>heading-aware chunks → MiniLM → FAISS"]

    CLS --> M1[(models/classify)]
    RET --> M2[(models/retrieve<br/>bi-encoder, cross-encoder,<br/>report vectors)]
    REC --> M3[(models/recommend)]
    RAGD --> M4[(models/rag<br/>7,237 passages)]
```

## Online: one request

```mermaid
flowchart LR
    U([Technician:<br/>family, variant, fault text]) --> P["1 preprocess<br/>clean · mask refs · normalise"]
    P --> C["2 classify<br/>DistilBERT multi-task"]
    P --> R["3 retrieve (same family)<br/>BM25 + MiniLM → RRF → cross-encoder, top 20"]
    R --> A["4 recommend<br/>similarity-weighted actions ·<br/>references by log-lift"]
    C --> D["5 passages<br/>FAISS over FAA docs<br/>(query = ATA title + fault)"]
    C --> E["Evidence<br/>labelled [Case n] / [doc p.N]"]
    A --> E
    R --> E
    D --> E
    E --> L["6 local LLM (Ollama, Qwen 2.5 7B)<br/>streamed, cited summary"]
    L --> V["citation validator<br/>labels + manual refs must be in evidence"]
    C --> UI
    A --> UI
    R --> UI
    V --> UI[["Streamlit UI / FastAPI JSON<br/>+ disclaimer"]]
    UI --> F[(SQLite feedback)]
    F -. "Phase 9: periodic update<br/>(index growth, condition head)" .-> M[(models)]
```

Steps 1–5 take ~0.1–0.25 s; the LLM summary streams in and completes in ~5 s on an RTX 4060 laptop GPU.

## Key design decisions (details in VIVA_NOTES.md)

| Decision | Why |
|---|---|
| Time-based split (≤2023 / 2024 / 2025) and cross-year dedup | The deployed system always predicts the future; random splits leak near-duplicate reports |
| Mask manual references, classify the *problem* text | References contain the ATA chapter; the action text is unknown when the fault is reported |
| Retrieval, not classification, drives recommendations | Similar cases carry the specific part and fix; a chapter label does not |
| Hybrid retrieval + reranking | BM25 catches exact part names; dense vectors catch paraphrases and typos |
| LLM sees only labelled evidence; output is validated | Keeps the summary grounded and auditable; no OEM procedures are generated |
| One `Advisor` for UI and API | No duplicated logic; models load once |
