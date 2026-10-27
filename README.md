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

Progress: see [docs/PROGRESS.md](docs/PROGRESS.md). Design reasoning: see [docs/VIVA_NOTES.md](docs/VIVA_NOTES.md).

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

```powershell
python download_data.py          # FAA SDR CSVs 1995-2026 + public FAA docs
python -m aerofix.check_env      # CUDA, libraries, Ollama
pytest                           # unit tests
python -m aerofix.data.build     # Phase 1: clean, dedup, split -> data/processed/ (~1 min first run)
python -m aerofix.eval.eda       # Phase 1: reports/eda.md + reports/figures/
python -c "import nltk; [nltk.download(p) for p in ('wordnet','omw-1.4','averaged_perceptron_tagger_eng','stopwords')]"
python -m aerofix.preprocess.build_norm   # Phase 2: fit normaliser, normalise corpus (~11 min, mostly lemmatisation)
python -m aerofix.eval.normalisation      # Phase 2: reports/normalisation.md (~10 min)
```

Two documents must be downloaded by hand from [drs.faa.gov](https://drs.faa.gov), because that site has no stable PDF links. Save them into `data/raw/docs/` as `TCDS_A28NM.pdf` (A320 family) and `TCDS_A16WE.pdf` (Boeing 737).

All settings (seed, paths, split years, hyperparameters) are in [configs/config.yaml](configs/config.yaml).

## Data

- **FAA Service Difficulty Reports (SDR)**, 1995–2026, filtered to the A320 family and the 737.
- **Public FAA reference documents**: AMT Handbooks FAA-H-8083-30B/31B/32B, AC 43.13-1B, and TCDS A28NM/A16WE.

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

_Filled in during Phase 10._
