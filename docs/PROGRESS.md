# Progress

| Phase | Status |
|---|---|
| 0. Setup | Done |
| 1. Data pipeline + EDA | Done |
| 2. Text normalisation | Done (abbrev.csv awaiting your review) |
| 3. Problem / action segmentation | Tooling done, waiting for annotation (400 reports) |
| 4. Classification | Not started |
| 5. Retrieval | Not started |
| 6. Recommenders | Not started |
| 7. RAG + LLM summary | Not started |
| 8. App | Not started |
| 9. Feedback loop | Not started |
| 10. Final evaluation | Not started |

## Phase 0: Setup (2026-10-28)

**Environment** (`python -m aerofix.check_env`)
- Python 3.12.0, torch 2.6.0+cu124, CUDA available on an RTX 4060 Laptop GPU (8.0 GB). bf16 supported; fp16 matmul OK.
- All 22 stack libraries import. Exact versions are in `requirements-lock.txt`.
- Ollama 0.40.11 is installed and both models answer.
  - `qwen2.5:7b-instruct` runs 100% on the GPU (4.8 GB). The first call took 58 s because it includes loading the model; a warm call generated 224 tokens in 4.6 s (~49 tokens/s).
  - The fallback `llama3.2:3b` also works.
- Known harmless warning: a `ResourceTracker ... _recursion_count` traceback at interpreter exit. It comes from `multiprocess` (a `datasets` dependency) on Python 3.12.0 and does not affect results.

**Tests:** `pytest` → 5 passed.

**Data** (`python download_data.py`)
- SDR CSVs: all 32 years (1995–2026) downloaded, 738 MB.
- **All 32 files share an identical 76-column header**, so the Phase 1 column mapping is one-to-one. Phase 1 still checks how fully each column is filled per year.
- Public docs (420 MB): FAA-H-8083-30B (General), 31B (Airframe), 32B (Powerplant), AC 43.13-1B with Change 1.
- TCDS A16WE (103 pages) and A28NM (91 pages) were saved manually from drs.faa.gov. Both are **scanned images with no text layer** (0 pages with text). Using them would need OCR, a new dependency to decide on in Phase 7. Until then they are excluded from RAG.

**Deviations from the brief**
- Python 3.12 instead of 3.11 (approved).
- Current single-volume "B" AMT handbooks instead of the older split volumes.
- `accelerate` is *not* installed yet. HF `Trainer` needs it, so approval will be requested before Phase 3.

## Phase 1: Data pipeline + EDA (2026-10-28)

Commands: `python -m aerofix.data.build` (~45 s with cached interim files; `--force` re-reads the CSVs), then `python -m aerofix.eval.eda`.
Outputs: `data/processed/sdr_a320_b737.parquet`, `reports/schema_mapping.md`, `reports/build_stats.json`, `reports/eda.md` and `reports/figures/`.

**Exit check passed.** Raw 2025 counts match the brief exactly: 67,652 rows, 22,935 B737, 12,415 A320, 38 chapters, 347 JASC codes. `build.py` asserts this on every run.

**Scope:** years 1995–2025. 2026 is a partial year and is left out for now (`build.years` in the config).

**Pipeline counts**
| Step | Rows |
|---|---|
| A320 + 737 rows, 1995–2025 | 567,350 |
| Dropped: invalid JASC code | 0 |
| Dropped: text under 3 words | 31 |
| Dropped: duplicates (same operator + same normalised text) | 27,311 |
| ...of which were later-year copies of an earlier report (would have leaked across splits) | 11,924 |
| **Final** | **540,008** |
| Train (≤2023) / val (2024) / test (2025) | 472,547 / 34,178 / 33,283 |

**Findings that affect later phases**
- **Schema:** all years share one 76-column header, so the column mapping is one-to-one.
  - Fill rates differ: `HowDiscoveredCode` is nearly empty before 2000, and `EngineModel`, `PartMake` and `ComponentName` are mostly empty in every year.
  - `PartCondition`, `PartName`, `JASCCode` and `Discrepancy` are ≥ 98% filled throughout.
- **Variants:** bare FAA codes `7378`/`7379` first appear in 2017/2018 and carry LEAP-1B engines, so they are mapped to MAX 8/9 (documented in the config). 12,839 rows have a generic `737`/`737*` code, mostly 2021–22; their variant is "unspecified".
- **Imbalance and drift:** ATA 53 is 46% of train but 35% of test, and the top 15 chapters cover 93.6% of train. Older reports are shorter (train median 24 words vs 31 in test) and cite fewer references (44% vs 76%).
  - So train and test distributions differ, which matters when reading Phase 4 results.
  - Option for Phase 4: also train on recent years only (e.g. 2015+) as an ablation.
- **Leakage:** 64% of 2025 reports that cite a reference cite one from the label's chapter (brief: ~63%).
  - The masking regex was extended for formats found in the data: `SRM B737-700 51-20-01`, `SRM-51-10-02`, `NDT PT 6-51-00-00`, `REFERENCE: TASK 25-66-01`.
  - A safety net also masks any leftover `NN-NN-NN` number.
  - Residual leak after masking: 0.24% of test rows (short forms like `CDL 32-10`, `A321AMM 72-15`).
- **Reference share:** 75.9% of 2025 reports cite at least one reference (brief: 74.6%).
- **Corrective-action verbs:** about 73% of test texts contain one. This is a rough proxy for the brief's ~76%; Phase 3 measures it properly.
- **Abbreviation candidates** (BS, FWD, STR, RBL, R&R, P/N, A/C ...) are listed in `reports/eda.md` §8 for Phase 2. "SRM"/"MM" also remain as bare words with no number. They carry a weak hint (structure), which the Phase 3 problem/action split mostly removes.

**Tests:** 42 passed.

## Phase 2: Text normalisation (2026-10-28)

Commands: `python -m aerofix.preprocess.build_norm`, then `python -m aerofix.eval.normalisation`.
Outputs:
- `models/normaliser.pkl`
- `data/processed/sdr_norm.parquet` (`text_norm`, `text_stem` and `text_lemma`, joined to the main table on `row_id`)
- `reports/normalisation.md`

Everything is fitted on train and measured on val; test was not used.

**Results (val)**
| Component | Result |
|---|---|
| Abbreviations | 123 expanding entries rewrite 4.3% of val tokens; 62% of reports contain at least one |
| LM perplexity | Laplace 1/2/3-gram: 862 / 409 / **2,834**; Kneser-Ney 1/2/3-gram: 864 / 77 / **41.2** |
| Spelling, synthetic typos (held-out half) | Edit distance only 96.7% → noisy channel (edit + trigram LM, penalty 15) **99.2%** |
| Spelling, real val text | Out-of-dictionary words 0.84% → 0.28% |
| Ablation, TF-IDF + LR on 100k train sample | Val macro-F1 0.853–0.855 for every variant: **no measurable effect** |

**Changes made after the first run**
- The spell corrector "fixed" valid rare words (LOADABLE → READABLE, DECELERATED → ACCELERATED). Real English words (WordNet) are now protected.
- The edit penalty is now tuned on half of the synthetic typos and reported on the other half. The original grid peaked at its edge (6.0); the wider grid levels off from 15.

**Known limitations**
- Some valid domain words that WordNet doesn't know still get "corrected": LOADABLE, UNRIVETED, RIBLETS, CEIL, SUBSTEP. These are listed in the report's spot-check table.
- Real-word errors (FORM for FROM) are not detected.
- Lemmatisation takes ~10 min for the corpus because of the POS tagger.

**Your action:** review `data/annotations/abbrev.csv` (153 rows: 123 to expand, 16 marked ambiguous, 14 kept as-is). Change an `expansion` or set `ambiguous` to yes/no, then rerun the two commands above.

## Phase 3: Problem / action segmentation (in progress)

**Done**
- Sample of 400 reports from 1995–2024: 280 train / 60 dev / 60 test, balanced by family and era (`data/annotations/seg_sample.parquet`).
- Annotation app, annotation guide, rule baseline, CRF, DistilBERT (own PyTorch loop), evaluation with token and span F1 plus a learning curve, and corpus application.
- Dry run on rule-generated "silver" labels to check the pipeline end to end (numbers not reported):
  - All three methods train, score and save.
  - Corpus inference takes ~16 min (BERT) or ~11 min (CRF).
- Tests: 64 passed.

**Waiting for:** annotation of the 400 reports (`streamlit run src/aerofix/segment/annotate_app.py`).
- Train/dev reports are pre-filled by the rules; test reports start blank (so the evaluation isn't biased towards the rules).
- Then run `python -m aerofix.segment.evaluate` and `python -m aerofix.segment.apply`.
