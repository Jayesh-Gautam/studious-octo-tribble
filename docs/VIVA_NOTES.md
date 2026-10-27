# Viva notes: design decisions and why

## Phase 0: Setup

**src layout + `pip install -e .`**
The code lives in `src/aerofix/` and is installed in editable mode. Tests and scripts then import the *installed* package, so they behave the same from any folder. This avoids `sys.path` hacks and catches packaging mistakes early.

**One config file (`configs/config.yaml`)**
Seeds, paths, split years, regexes and hyperparameters live in one place. Any result can be traced to a config, and nothing depends on where the project sits on disk. `load_config()` turns relative paths into absolute ones under the project root.

**Fixed seeds (`set_seed`)**
Python `random`, NumPy and PyTorch (CPU + CUDA) are all seeded from `seed: 42`. GPU kernels can still differ slightly between runs, so reported numbers are from single seeded runs. Small differences are noise, not signal.

**Time-based split (train ≤ 2023, val 2024, test 2025)**
In deployment the model always predicts *future* reports from *past* ones. A random split would leak information, because the same operator writes near-identical reports in the same month. Splitting by year matches real use and is the stricter test.

**Python 3.12 instead of 3.11**
3.12 was already installed, and every library in the stack ships 3.12 wheels (PyTorch CUDA, faiss-cpu, sklearn-crfsuite). Changing version would add setup work for no benefit.

**PyTorch installed separately from `requirements.txt`**
The default PyPI `torch` wheel on Windows is CPU-only. The CUDA build comes from PyTorch's own index (`cu124`). Keeping it as a separate, documented step makes sure the GPU build is used. `requirements-lock.txt` records the exact versions installed.

**faiss-cpu over chromadb**
FAISS is a single library doing exact or approximate vector search. Our corpus (~tens of thousands of reports plus a few thousand doc chunks) fits in memory, so we don't need a database server. FAISS is also easier to explain: it is just an index over a matrix of embeddings.

**FAA documents: current "B" handbook editions only**
FAA-H-8083-30B/31B/32B replace the older split volumes. Indexing both editions would put near-duplicate passages in the RAG index and crowd out other results.

## Phase 1: Data pipeline

**Read only the needed columns, as strings, and filter at once.** Each CSV has 76 columns and ~60k rows. Reading 22 columns with `dtype=str` avoids pandas guessing types (part numbers like `0012` would lose zeros) and keeps memory low. Per-year Parquet caches make re-runs take seconds.

**UTF-8, checked rather than assumed.** All 32 files decode as strict UTF-8, so they are read that way. Latin-1 would silently garble the ~15k non-ASCII characters in later years.

**Deduplicate before splitting, keep the earliest copy.** The same operator often files the same text again in a later year. If the 2023 copy is in train and the 2025 copy in test, the model is tested on a report it has already seen. 11,924 such cross-year copies were removed. The dedup key ignores punctuation and spacing ("R/H" = "R H").

**Split on file year, not a random split.** Every `DifficultyDate` falls inside its file's year (0 mismatches), so file year is a clean time stamp.

**Why mask references (label leakage).** "PER AMM 52-10-00" literally contains the answer (ATA 52). 64% of 2025 reports with a reference cite the label's chapter. A model trained on raw text would learn to read reference numbers instead of understanding the fault. We keep the references in a `refs` column (used by the Phase 6 recommender) and train on `text_masked`. Phase 4 measures F1 with and without masking to show the effect.

**Why the regex grew.** The brief's regex missed real formats (`M/M 33-51-06`, chained `SRM 51-10-02; 51-40-02`, `SRM B737-700 51-20-01`). Each new format has a unit test (`tests/test_refs.py`), so a regex change can't silently break an older case. A final safety net masks any leftover `NN-NN-NN` number without adding it to `refs`, because a few could be dates.

**Boilerplate kept as a feature.** "AIRCRAFT WAS GROUNDED:" says something about severity, not about which system failed. It is stripped from the text (so it doesn't dominate similarity) and kept in `context`.

**MAX variant mapping is evidence-based.** Bare codes `7378`/`7379` appear only from 2017/2018 (MAX entry into service), and where the engine field is filled it is LEAP-1B (MAX engine) rather than CFM56-7B (NG engine).

**Why top-K chapters come from train only.** Choosing classes by looking at the test year would be a (mild) form of tuning on test.
