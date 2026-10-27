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
