"""BiLSTM with self-trained Word2Vec: python -m aerofix.classify.bilstm

1. Word2Vec (gensim, skip-gram) trained on TRAIN reports only (full normalised
   text from Phase 2, so the vectors also learn from action context). Skip-gram
   learns a vector per word such that words appearing in similar contexts
   (CORROSION / CORRODED, INOP / INOPERATIVE) end up close together.
2. Embedding layer initialised with those vectors (and fine-tuned).
3. 2-layer bidirectional LSTM reads the problem text left->right and right->left;
   max-pooling over time keeps, per feature, its strongest value anywhere in the
   text (padding masked out), giving a fixed-size vector.
4. Dropout + three linear heads (ata2, ata4, condition); training loop in neural.py.
Input is ``problem_norm`` (Phase 2 abbreviations + spelling on the problem text):
word-level embeddings suffer most from spelling variants, so this is where
normalisation should matter.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

from aerofix.classify import neural
from aerofix.classify.data import TARGETS, LabelSpace, class_weights, load_frame, save_preds
from aerofix.config import load_config
from aerofix.utils import get_logger, set_seed

log = get_logger(__name__)
PAD, UNK = 0, 1
MAX_TOKENS = 100


class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, emb: np.ndarray, hidden: int, layers: int, dropout: float, n_classes: dict[str, int]):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=False, padding_idx=PAD)
        self.lstm = torch.nn.LSTM(emb.shape[1], hidden, num_layers=layers, bidirectional=True, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleDict({t: torch.nn.Linear(2 * hidden, n) for t, n in n_classes.items()})

    def forward(self, ids, mask):
        h, _ = self.lstm(self.drop(self.emb(ids)))
        h = h.masked_fill(mask.unsqueeze(-1) == 0, -1e4)  # padding can never win the max
        pooled = self.drop(h.max(dim=1).values)
        return {t: head(pooled) for t, head in self.heads.items()}


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    bc = cfg["classify"]["bilstm"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg["paths"]["models"] / "classify" / "bilstm"
    df = load_frame(cfg)
    norm = pd.read_parquet(cfg["paths"]["processed"] / "sdr_norm.parquet", columns=["row_id", "text_norm"])
    df = df.merge(norm, on="row_id", how="left")
    labels = LabelSpace.load(cfg["paths"]["models"] / "classify" / "labels.json")
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))

    t = time.perf_counter()
    sentences = [s.split() for s in train.text_norm.fillna("")]
    w2v = Word2Vec(sentences, vector_size=bc["w2v_dim"], window=bc["w2v_window"], min_count=bc["w2v_min_count"],
                   sg=1, workers=8, epochs=bc["w2v_epochs"], seed=cfg["seed"])
    log.info("word2vec: %d words in %.0f s; neighbours of CORROSION: %s", len(w2v.wv), time.perf_counter() - t,
             [w for w, _ in w2v.wv.most_similar("CORROSION", topn=5)])
    out.mkdir(parents=True, exist_ok=True)
    w2v.save(str(out / "word2vec.model"))

    vocab = {w: i + 2 for i, w in enumerate(w2v.wv.index_to_key)}
    emb = np.zeros((len(vocab) + 2, bc["w2v_dim"]), dtype=np.float32)
    emb[UNK] = np.random.default_rng(cfg["seed"]).normal(0, 0.1, bc["w2v_dim"])
    emb[2:] = w2v.wv.vectors

    def to_ids(texts):
        return [[vocab.get(w, UNK) for w in s.split()[:MAX_TOKENS]] or [UNK] for s in texts.fillna("")]

    seqs = {s: to_ids(d.problem_norm) for s, d in (("train", train), ("val", val), ("test", test))}
    oov = np.mean([i == UNK for s in seqs["val"] for i in s])
    log.info("val problem tokens not in word2vec vocab: %.2f%%", 100 * oov)
    ys = {s: {tg: labels.encode(tg, d[tg]) for tg in TARGETS} for s, d in (("train", train), ("val", val), ("test", test))}
    n_classes = {tg: len(labels.names(tg)) for tg in TARGETS}
    weights = {tg: class_weights(ys["train"][tg], n_classes[tg], cfg["classify"]["class_weight_power"]) for tg in TARGETS}

    model = BiLSTMClassifier(emb, bc["hidden"], bc["layers"], bc["dropout"], n_classes).to(device)
    make_opt = lambda m, total, warmup: (torch.optim.Adam(m.parameters(), lr=bc["lr"]), None)
    model, history = neural.train(model, seqs["train"], ys["train"], seqs["val"], ys["val"], weights, pad_id=PAD,
                                  device=device, epochs=bc["epochs"], batch_size=bc["batch_size"],
                                  make_optimizer=make_opt, patience=bc["patience"], seed=cfg["seed"], log=log.info)
    for split, d in (("val", val), ("test", test)):
        save_preds(out, split, d.row_id.values, neural.predict(model, seqs[split], PAD, device))
    neural.save_run(out, model, {"model": "bilstm", "text_col": "problem_norm", "vocab_size": len(vocab) + 2,
                                 "w2v_oov_val": round(float(oov), 4), **bc}, history)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
