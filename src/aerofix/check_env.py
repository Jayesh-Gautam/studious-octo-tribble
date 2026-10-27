"""Environment check: python -m aerofix.check_env

Confirms the GPU is usable for training, every library in the stack imports, and
the local Ollama LLM answers. Exits with code 1 if CUDA is unavailable, because
Phases 3-5 cannot train in reasonable time on CPU.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
import time

from aerofix.config import load_config

# import name -> pip name (they differ for a few packages)
STACK = {
    "pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy", "yaml": "pyyaml",
    "sklearn": "scikit-learn", "nltk": "nltk", "sklearn_crfsuite": "sklearn-crfsuite",
    "gensim": "gensim", "rank_bm25": "rank-bm25", "lightgbm": "lightgbm",
    "transformers": "transformers", "datasets": "datasets",
    "sentence_transformers": "sentence-transformers", "faiss": "faiss-cpu",
    "pymupdf": "pymupdf", "ollama": "ollama", "streamlit": "streamlit",
    "fastapi": "fastapi", "uvicorn": "uvicorn", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "pytest": "pytest",
}


def check_torch() -> bool:
    import torch

    print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("  CUDA available: False  <-- install the cu124 wheel (see README)")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"  CUDA available: True | {props.name} | {props.total_memory / 2**30:.1f} GB VRAM")
    print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
    # Tiny fp16 matmul: proves kernels actually run, not just that a driver exists.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    (a @ a).sum().item()
    print(f"  fp16 1024x1024 matmul OK ({(time.perf_counter() - t) * 1000:.1f} ms)")
    return True


def check_imports() -> list[str]:
    missing = []
    for mod, pip_name in STACK.items():
        try:
            importlib.import_module(mod)
            # Installed-distribution version: works even for modules without __version__.
            print(f"  {pip_name:<22} {importlib.metadata.version(pip_name)}")
        except Exception as e:  # report every failure, don't stop at the first
            print(f"  {pip_name:<22} MISSING ({e.__class__.__name__}: {e})")
            missing.append(pip_name)
    return missing


def check_ollama(cfg: dict) -> bool:
    import ollama

    client = ollama.Client(host=cfg["llm"]["host"])
    for model in (cfg["llm"]["model"], cfg["llm"]["fallback_model"]):
        try:
            t = time.perf_counter()
            r = client.chat(model=model, messages=[{"role": "user", "content": "Say hello in five words."}])
            print(f"  {model}: {r['message']['content'].strip()!r} ({time.perf_counter() - t:.1f} s)")
        except Exception as e:
            print(f"  {model}: FAILED ({e.__class__.__name__}: {e})")
            return False
    return True


def main() -> None:
    cfg = load_config()
    print(f"Python {sys.version.split()[0]} on {platform.platform()}\n")
    cuda_ok = check_torch()
    print("\nLibraries:")
    missing = check_imports()
    print("\nOllama:")
    ollama_ok = check_ollama(cfg)

    print("\nSummary:")
    print(f"  CUDA:    {'OK' if cuda_ok else 'FAIL'}")
    print(f"  Imports: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    print(f"  Ollama:  {'OK' if ollama_ok else 'FAIL (is Ollama running and are the models pulled?)'}")
    sys.exit(0 if cuda_ok and not missing else 1)


if __name__ == "__main__":
    main()
