"""Multi-task transformer classifier: python -m aerofix.classify.transformer --model distilbert

Encoder (DistilBERT or RoBERTa-base) -> representation of the first token
([CLS] / <s>) -> dropout -> three linear heads (ata2, ata4, condition).
Fine-tuned end to end with AdamW, linear warm-up + decay, fp16, weighted
cross-entropy, early stopping on val ata2 macro-F1 (see neural.py).

Text is lower-cased for both models: DistilBERT-uncased requires it, and RoBERTa's
byte-level BPE was trained on mostly lower-case text, so ALL-CAPS reports would be
split into many rare pieces.

Options for the ablation runs:
  --text-col text | text_masked   leakage ablation on the full report
  --train-family A320 | B737      cross-family transfer (train size capped to the smaller family)
  --epochs N                      override the configured epoch count
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from aerofix.classify import neural
from aerofix.classify.data import TARGETS, LabelSpace, class_weights, load_frame, save_preds
from aerofix.config import load_config
from aerofix.utils import get_logger, set_seed

log = get_logger(__name__)


class MultiTaskTransformer(torch.nn.Module):
    def __init__(self, name: str, n_classes: dict[str, int], dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        hid = self.encoder.config.hidden_size
        self.drop = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleDict({t: torch.nn.Linear(hid, n) for t, n in n_classes.items()})

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        h = self.drop(h)
        return {t: head(h) for t, head in self.heads.items()}


def encode(tok, texts, max_len: int) -> list[list[int]]:
    enc = tok([t.lower() for t in texts], truncation=True, max_length=max_len)
    return enc["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["distilbert", "roberta"], default="distilbert")
    ap.add_argument("--text-col", help="input column (default from config: problem_masked)")
    ap.add_argument("--train-family", choices=["A320", "B737"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name", help="output dir name under models/classify/")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tc = cfg["classify"]["transformer"]
    mcfg = tc["models"][args.model]
    col = args.text_col or cfg["classify"]["text_col"]
    name = args.name or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg["paths"]["models"] / "classify" / name

    df = load_frame(cfg)
    labels = LabelSpace.load(cfg["paths"]["models"] / "classify" / "labels.json")
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    if args.train_family:
        n = min((train.family == "A320").sum(), (train.family == "B737").sum())
        train = train[train.family == args.train_family].sample(n, random_state=cfg["seed"])
    log.info("%s | input %s | train %d rows | device %s", name, col, len(train), device)

    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    seqs = {s: encode(tok, d[col].fillna(""), tc["max_seq_len"]) for s, d in (("train", train), ("val", val), ("test", test))}
    ys = {s: {t: labels.encode(t, d[t]) for t in TARGETS} for s, d in (("train", train), ("val", val), ("test", test))}
    n_classes = {t: len(labels.names(t)) for t in TARGETS}
    weights = {t: class_weights(ys["train"][t], n_classes[t], cfg["classify"]["class_weight_power"]) for t in TARGETS}

    model = MultiTaskTransformer(mcfg["name"], n_classes).to(device)

    def make_optimizer(m, total, warmup):
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [{"params": [p for n_, p in m.named_parameters() if not any(k in n_ for k in no_decay)], "weight_decay": 0.01},
                  {"params": [p for n_, p in m.named_parameters() if any(k in n_ for k in no_decay)], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=tc["lr"])
        return opt, get_linear_schedule_with_warmup(opt, warmup, total)

    model, history = neural.train(
        model, seqs["train"], ys["train"], seqs["val"], ys["val"], weights, pad_id=tok.pad_token_id, device=device,
        epochs=args.epochs or mcfg["epochs"], batch_size=tc["batch_size"], make_optimizer=make_optimizer,
        warmup_frac=tc["warmup_frac"], patience=tc["patience"], seed=cfg["seed"], log=log.info)

    for split, d in (("val", val), ("test", test)):
        save_preds(out, split, d.row_id.values, neural.predict(model, seqs[split], tok.pad_token_id, device))
    neural.save_run(out, model, {"model": mcfg["name"], "text_col": col, "train_family": args.train_family,
                                 "train_rows": len(train), "max_seq_len": tc["max_seq_len"]}, history)
    tok.save_pretrained(out / "tokenizer")
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
