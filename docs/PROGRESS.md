# Progress

| Phase | Status |
|---|---|
| 0. Setup | Done |
| 1. Data pipeline + EDA | Done |
| 2. Text normalisation | Done (abbrev.csv awaiting your review) |
| 3. Problem / action segmentation | Done (silver LLM labels; human check of test optional) |
| 4. Classification | Done |
| 5. Retrieval | Done |
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
- `accelerate` is not installed: Phase 3 uses our own PyTorch training loop instead of HF `Trainer` (your choice), so it is not needed.

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

## Phase 3: Problem / action segmentation (2026-10-28)

Commands:
1. `python -m aerofix.segment.sample`
2. `python -m aerofix.segment.llm_label` (and/or the human annotation tool)
3. `python -m aerofix.segment.evaluate`
4. `python -m aerofix.segment.apply`

Outputs: `reports/segmentation.md`, `models/segment/` and `data/processed/sdr_segments.parquet`.

**Annotation (change of plan):** 399 of 400 reports were labelled by the local LLM (Qwen 2.5 7B, chunk level, schema-constrained JSON, deterministic REF rules), and 1 by a human. These are *silver* labels: test scores measure agreement with the LLM. Human labels override LLM labels whenever added.

**Results (60 test reports)**
| Method | Token macro-F1 | Span micro-F1 |
|---|---|---|
| Rules (baseline) | 0.508 | 0.194 |
| **CRF** (chosen on dev) | 0.781 | **0.681** |
| DistilBERT (own loop) | **0.798** | 0.580 |

- CRF selected (dev span F1 0.71 vs 0.61): its transition weights give clean spans, while BERT's per-token softmax fragments them.
- Learning curve: the CRF is fairly flat (0.63 → 0.69 span F1 from 70 to 280 reports); BERT is noisier.

**Corpus output (CRF on 540,008 reports):**
- Median problem text: 13 words.
- Has an action part: 94–97% per split.
- No PROBLEM tokens found (whole masked report used instead): 1.2% train, 2.3% val, 1.7% test.

**Optional, recommended for the viva:** hand-label the 60 test reports in the annotation tool (≈30 min). Human labels automatically replace the LLM ones; then rerun `segment.evaluate`.

**Tests:** 67 passed.

## Phase 4: Classification (2026-10-28)

Commands:
1. `python -m aerofix.classify.prepare`
2. `python -m aerofix.classify.tfidf`
3. `python -m aerofix.classify.bilstm`
4. `python -m aerofix.classify.transformer --model distilbert` (and `--model roberta`, plus the ablation runs in the README)
5. `python -m aerofix.classify.report`

Outputs: `reports/classification.md`, `reports/figures/confusion_*.png` and `models/classify/`.

**Input:** `problem_masked`. **Targets:** ata2 (top 15 + OTHER), ata4 (225 JASC codes + OTHER), condition (top 20 raw codes + OTHER). Tuned on val; test scored once with frozen models.

**ATA2 results (macro-F1 / accuracy)**
| Model | Val 2024 | Test 2025 |
|---|---|---|
| Majority class | 0.036 / 0.41 | 0.032 / 0.35 |
| TF-IDF + LR | 0.822 / 0.870 | 0.820 / 0.892 |
| BiLSTM + Word2Vec | 0.840 / 0.893 | 0.847 / 0.913 |
| **DistilBERT, multi-task** (selected on val) | **0.848 / 0.898** | 0.859 / 0.922 |
| RoBERTa-base, multi-task | 0.844 / 0.897 | 0.860 / 0.922 |

- Top-3 accuracy is ≈0.99 for all neural models.
- ata4 macro-F1: ~0.39 (225 classes).
- Condition macro-F1: ~0.60. The condition vocabulary drifts: OTHER is 15% of train but 24% of test.

**Findings**
- **Leakage:** full report unmasked → masked costs 0.9 points (DistilBERT) and 0.9 points (TF-IDF, tokenised char n-grams). The realistic problem-only input scores lower again, because the action text carries information the technician doesn't have yet.
- **RoBERTa ≈ DistilBERT:** RoBERTa has twice the parameters and twice the training time for no gain on val.
- **Cross-family:** training on the other family costs 5–7 macro-F1 points (DistilBERT, equal train size).
- **Normalisation:** no effect beyond tokenisation (control row). An earlier "+2 points" turned out to come from dropping punctuation before the character n-grams. That is now part of the TF-IDF pipeline.
- **Recent years only (2015–2023):** +0.7 points for TF-IDF on an equal-size sample.
- **Main confusions:** 49 APU ↔ 21 air conditioning (fume events), 72 engine → OTHER, 52 doors ↔ 25 equipment/slides.

**Notes:** the ata4 head of TF-IDF uses SGD logistic regression, because L-BFGS needed 10 GB for 226 × 242k weights. **Tests:** 72 passed.

## Phase 5: Retrieval of similar past cases (2026-10-28)

Command: `python -m aerofix.retrieve.evaluate` (~25 min: encode corpus, fine-tune bi-encoder + cross-encoder, val + test stages, LLM check).
Outputs: `reports/retrieval.md` and `models/retrieve/` (`biencoder/`, `crossencoder/`, `vectors_ft.npy`, `best.json`).

**Setup**
- Index text and queries: `problem_masked`, with search restricted to the same aircraft family.
- Val stage: 1,000 queries from 2024 against the train index; used to choose the BM25 variant and the app's method.
- Test stage: 2,000 queries from 2025 against train + val.
- Proxy relevance: same JASC code + same part condition.

**Test results (NDCG@5 / P@5 / ms per query)**
| Method | NDCG@5 | P@5 | ms/query |
|---|---|---|---|
| BM25 raw → + stem (best BM25 on test) | 0.587 → 0.595 | 0.359 → 0.370 | ~10 |
| Average Word2Vec | 0.607 | 0.379 | 0.5 |
| MiniLM zero-shot | 0.616 | 0.403 | 0.6 |
| **MiniLM fine-tuned** (MNRL with hard negatives) | **0.634** | **0.430** | 0.6 |
| Hybrid RRF (BM25 + lemma + fine-tuned) | 0.625 | 0.417 | 10 |
| Hybrid + cross-encoder rerank (best on val: 0.616 vs 0.606) | 0.626 | 0.427 | 26 |

**Findings**
- **Normalisation helps BM25** (+0.5–1 NDCG point on val and test): the first place the Phase 2 normalisation measurably pays off, as expected for keyword matching.
- **Fine-tuning the bi-encoder:** +1.8 NDCG points over zero-shot on test.
- **Reranking and hybrid** helped on val but not on test. Differences of ~1 point are within noise (SE ≈ 0.01 at 2,000 queries).
- **The LLM check (50 test queries, top 5)** rates BM25 highest (P@5 0.56 vs 0.50 fine-tuned, 0.53 hybrid + rerank). LLM and proxy agree on only 63% of pairs. The fine-tuned model is trained on the proxy definition, so proxy metrics favour it, while a reader-like judge prefers lexical matches. 50 queries is not conclusive, but proxy-only evaluation clearly flatters the trained model. Human judgement of a sample is the right next step.
- **App choice:** hybrid + cross-encoder rerank (protocol: best on val). It combines lexical and semantic matches, at 26 ms per query.

**Tests:** 78 passed.
