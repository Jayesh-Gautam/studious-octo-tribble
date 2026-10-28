# AeroFix: Maintenance Fault Advisor (A320 family / Boeing 737)

AeroFix is a final-year NLP course project. A line technician selects an aircraft and types a fault in free text, for example `"L1 door damper leaking, emer cyl press low"`. AeroFix returns:

- the predicted ATA system
- the predicted part condition
- the most common corrective actions in similar past cases
- the manual references technicians cited for this kind of fault
- similar past FAA Service Difficulty Reports
- related public FAA document passages
- a short cited summary written by a local LLM

> **Disclaimer:** AeroFix is decision support only. Always follow approved OEM/AMM data. No proprietary Airbus/Boeing manual content is used. AeroFix only points to manual *references* that technicians cited in public FAA reports.

**Results:** [reports/RESULTS.md](reports/RESULTS.md) · **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Design reasoning:** [docs/VIVA_NOTES.md](docs/VIVA_NOTES.md) · **Progress log:** [docs/PROGRESS.md](docs/PROGRESS.md)

## Results at a glance (2025 test year, never used for tuning)

| Component | Baseline | AeroFix |
|---|---|---|
| ATA chapter (macro-F1) | majority 0.03 · TF-IDF + LR 0.82 | **DistilBERT 0.86**, top-3 accuracy 0.99 |
| Part condition (macro-F1) | majority 0.02 | **0.61** |
| Similar-case retrieval (NDCG@5) | BM25 0.59 | **fine-tuned MiniLM 0.63** |
| Corrective action (hit@3) | most frequent overall 0.12 | **retrieval vote 0.46** |
| Manual reference (hit@3) | most cited overall 0.23 | **retrieval 0.74** |
| Cited LLM summary | – | 100% cite, 97% of citations valid, 0 invented references |

Full tables, ablations (reference masking, normalisation, stemming vs lemmatisation, cross-family), the feedback learning curve, 20 analysed failure cases and the limitations are in [reports/RESULTS.md](reports/RESULTS.md).

## Setup (Windows, Python 3.12, NVIDIA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1. PyTorch with CUDA (the PyPI default is CPU-only on Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2. Everything else (exact versions: requirements-lock.txt)
pip install -r requirements.txt
pip install -e .
```

**Local LLM:** install [Ollama](https://ollama.com/download), then:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull llama3.2:3b
```

## Reproduce

One command regenerates `reports/RESULTS.md` (all tables and figures) from the saved models in about a minute; `--full` re-runs every phase from the raw data first (~4–5 h on an RTX 4060 laptop, needs Ollama running):

```powershell
python download_data.py
python -m aerofix.eval.final --full
```

Or step by step:

```powershell
python download_data.py          # FAA SDR CSVs 1995-2026 + public FAA docs
python -m aerofix.check_env      # CUDA, libraries, Ollama
pytest                           # unit tests
python -m aerofix.data.build     # Phase 1: clean, dedup, split -> data/processed/ (~1 min first run)
python -m aerofix.eval.eda       # Phase 1: reports/eda.md + reports/figures/
python -c "import nltk; [nltk.download(p) for p in ('wordnet','omw-1.4','averaged_perceptron_tagger_eng','stopwords')]"
python -m aerofix.preprocess.build_norm   # Phase 2: fit normaliser, normalise corpus (~11 min, mostly lemmatisation)
python -m aerofix.eval.normalisation      # Phase 2: reports/normalisation.md (~10 min)
python -m aerofix.segment.sample          # Phase 3: draw the 400 reports to annotate (once)
python -m aerofix.segment.llm_label       # Phase 3: LLM (Ollama) silver labels for reports without a human label (~6 min)
streamlit run src/aerofix/segment/annotate_app.py   # Phase 3: annotate (see docs/ANNOTATION_GUIDE.md)
python -m aerofix.segment.evaluate        # Phase 3: rules vs CRF vs DistilBERT -> reports/segmentation.md
python -m aerofix.segment.apply           # Phase 3: segment the corpus with the best model (~15 min)
python -m aerofix.classify.prepare        # Phase 4: normalise problem texts (~5 min)
python -m aerofix.classify.tfidf          # Phase 4: TF-IDF + LR models and ablations (~1 h, CPU)
python -m aerofix.classify.bilstm         # Phase 4: Word2Vec + BiLSTM (~7 min, GPU)
python -m aerofix.classify.transformer --model distilbert   # Phase 4 (~35 min, GPU)
python -m aerofix.classify.transformer --model roberta      # Phase 4 (~50 min, GPU)
python -m aerofix.classify.transformer --model distilbert --text-col text --epochs 1 --name distilbert_full_unmasked
python -m aerofix.classify.transformer --model distilbert --text-col text_masked --epochs 1 --name distilbert_full_masked
python -m aerofix.classify.transformer --model distilbert --train-family A320 --epochs 2 --name distilbert_A320
python -m aerofix.classify.transformer --model distilbert --train-family B737 --epochs 2 --name distilbert_B737
python -m aerofix.classify.report        # Phase 4: reports/classification.md
python -m aerofix.retrieve.evaluate      # Phase 5: train bi-/cross-encoder, evaluate retrieval (~25 min, GPU + Ollama)
python -m aerofix.recommend.evaluate     # Phase 6: canonical actions + action/reference recommenders (~10 min)
python -m aerofix.rag.docs               # Phase 7: parse FAA PDFs into 7k cited passages (~2 min)
python -m aerofix.rag.evaluate           # Phase 7: 50 cited LLM summaries + checks (~6 min; add --model llama3.2:3b to compare)
python -m aerofix.feedback.simulate      # Phase 9: 3 simulated feedback rounds + learning curve (~15 min)
python -m aerofix.eval.final             # Phase 10: reports/RESULTS.md
```

Two documents must be downloaded by hand from [drs.faa.gov](https://drs.faa.gov), because that site has no stable PDF links. Save them into `data/raw/docs/` as `TCDS_A28NM.pdf` (A320 family) and `TCDS_A16WE.pdf` (Boeing 737).

All settings (seed, paths, split years, hyperparameters) are in [configs/config.yaml](configs/config.yaml).

## Data

- **FAA Service Difficulty Reports (SDR)**, 1995–2025 (2026 is downloaded but not used yet: partial year), filtered to the A320 family and the 737: 540,008 reports after cleaning and deduplication.
- **Public FAA reference documents**: AMT Handbooks FAA-H-8083-30B/31B/32B and AC 43.13-1B (7,237 passages). The TCDS A28NM/A16WE PDFs are scanned images without text and are not indexed.

Splits are by time: train ≤ 2023, validation 2024, test 2025 (held out).

## Project layout

```
configs/      config.yaml
data/         raw/ (downloads), interim/, processed/, annotations/ (hand labels, tracked)
src/aerofix/  data, preprocess, segment, classify, retrieve, recommend, rag, feedback, eval, app
tests/        pytest
docs/         PROGRESS, VIVA_NOTES, ARCHITECTURE
reports/      figures and result tables
```

## Syllabus mapping

| Syllabus topic | Where in AeroFix | Code | Evidence |
|---|---|---|---|
| Text preprocessing (cleaning, tokenisation, regex) | Cleaning, boilerplate stripping, manual-reference extraction/masking, domain tokeniser that keeps `R&R`, `P/N` | `preprocess/clean.py`, `refs.py`, `tokenize.py` | `reports/eda.md`, `tests/test_refs.py` |
| Abbreviations and spelling correction | Mined + reviewed abbreviation dictionary; noisy-channel corrector (edit distance × LM) | `preprocess/abbrev.py`, `spell.py` | `reports/normalisation.md` §1, §3 |
| Stemming vs lemmatisation (NLTK) | Porter stemmer vs WordNet lemmatiser with POS tags, compared for classification and BM25 | `preprocess/normalise.py` | `RESULTS.md` §3c, §6 |
| N-gram language model | Unigram/bigram/trigram, Laplace vs interpolated Kneser-Ney, written from scratch; perplexity | `preprocess/ngram_lm.py` | `reports/normalisation.md` §2 |
| Vectorisation: TF-IDF, BM25 | Word + character n-gram TF-IDF classifier; sparse BM25 retrieval | `classify/tfidf.py`, `retrieve/index.py` | `RESULTS.md` §3, §6 |
| Word embeddings (Word2Vec) | Skip-gram trained on the corpus; BiLSTM input; average-vector retrieval | `classify/bilstm.py`, `retrieve/evaluate.py` | `reports/classification.md` |
| RNN / BiLSTM | 2-layer BiLSTM with max-pooling, multi-task heads | `classify/bilstm.py` | `RESULTS.md` §3 |
| Sequence labelling (CRF, BIO) | Problem/action/reference segmentation: rules vs CRF vs DistilBERT token classifier | `segment/` | `reports/segmentation.md` |
| Transformers / BERT classification | Fine-tuned DistilBERT and RoBERTa, multi-task (ATA, JASC, condition), own training loop | `classify/transformer.py`, `neural.py` | `RESULTS.md` §3 |
| Information retrieval | BM25, bi-encoder (MNRL fine-tuning), cross-encoder reranking, reciprocal rank fusion, FAISS | `retrieve/` | `reports/retrieval.md` |
| Clustering | Agglomerative clustering of action objects (embeddings, cosine) | `recommend/actions.py` | `reports/recommend.md` |
| Retrieval-augmented generation (RAG) | PDF chunking, passage index, local LLM with citations and a validator | `rag/` | `reports/rag.md` |
| Evaluation methodology | Time-based split, leakage ablation, macro-F1, span F1, NDCG, proxy vs LLM judgement, learning curve | `eval/final.py` | `reports/RESULTS.md` |

## Run the app

```powershell
ollama serve                                         # if Ollama is not already running
streamlit run src/aerofix/app/streamlit_app.py        # UI at http://localhost:8501 (~35 s model load)
uvicorn aerofix.app.api:app --port 8000               # optional API: POST /advise, POST /feedback, docs at /docs
```
