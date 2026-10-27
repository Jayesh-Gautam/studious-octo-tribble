# Progress

| Phase | Status |
|---|---|
| 0. Setup | Done |
| 1. Data pipeline + EDA | Not started |
| 2. Text normalisation | Not started |
| 3. Problem / action segmentation | Not started |
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
- TCDS A28NM and A16WE: optional, and only needed in Phase 7. drs.faa.gov is a JavaScript app, so they must be saved from a browser (see README).

**Deviations from the brief**
- Python 3.12 instead of 3.11 (approved).
- Current single-volume "B" AMT handbooks instead of the older split volumes.
- `accelerate` is *not* installed yet. HF `Trainer` needs it, so approval will be requested before Phase 3.
