"""Fine-tuned DistilBERT token classifier, trained with a plain PyTorch loop.

Words -> WordPiece sub-words: "DELAMINATED" may become "del ##ami ##nated". Only the
FIRST sub-word of each word carries the word's BIO tag; the others get -100, which
the cross-entropy loss ignores. At prediction time we read the tag at each word's
first sub-word.

Training loop (what HF Trainer would hide):
  AdamW (weight decay 0.01), linear warm-up for 10% of steps then linear decay,
  fp16 autocast + GradScaler on the GPU, gradient clipping at 1.0, and early
  stopping on dev token macro-F1 (best weights kept in memory).
Reports longer than the window are split into windows of ``window`` words.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from aerofix.segment.common import BIO, from_bio, to_bio

TAG2ID = {t: i for i, t in enumerate(BIO)}


class BertSegmenter:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 256, window: int = 150,
                 device: str | None = None):
        self.model_name, self.max_len, self.window = model_name, max_len, window
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(BIO), id2label=dict(enumerate(BIO)), label2id=TAG2ID).to(self.device)

    # ---------- encoding ----------
    def _windows(self, tokens: list[str]) -> list[tuple[int, int]]:
        return [(s, min(s + self.window, len(tokens))) for s in range(0, max(len(tokens), 1), self.window)]

    def _encode(self, words: list[str], tags: list[str] | None):
        enc = self.tok([w.lower() for w in words], is_split_into_words=True, truncation=True,
                       max_length=self.max_len)
        if tags is not None:
            labels, prev = [], None
            for wid in enc.word_ids():
                labels.append(-100 if wid is None or wid == prev else TAG2ID[tags[wid]])
                prev = wid
            enc["labels"] = labels
        return enc

    def _batches(self, encs: list, batch_size: int, shuffle: bool):
        def collate(items):
            n = max(len(e["input_ids"]) for e in items)
            pad = lambda key, v: [e[key] + [v] * (n - len(e[key])) for e in items]
            out = {"input_ids": torch.tensor(pad("input_ids", self.tok.pad_token_id)),
                   "attention_mask": torch.tensor(pad("attention_mask", 0))}
            if "labels" in items[0]:
                out["labels"] = torch.tensor(pad("labels", -100))
            return out
        return DataLoader([dict(e) for e in encs], batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    # ---------- training ----------
    def fit(self, train, dev, epochs: int = 20, lr: float = 5e-5, batch_size: int = 16, patience: int = 5,
            seed: int = 42, score_fn=None, log=print) -> "BertSegmenter":
        torch.manual_seed(seed)
        encs = []
        for toks, labs in train:
            bio = to_bio(labs)
            for s, e in self._windows(toks):
                encs.append(self._encode(toks[s:e], bio[s:e]))
        loader = self._batches(encs, batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total = epochs * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best, best_state, bad = -math.inf, None, 0
        for ep in range(1, epochs + 1):
            self.model.train()
            losses = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**batch).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            dev_score = score_fn([lab for _, lab in dev], self.predict([t for t, _ in dev]))
            log(f"epoch {ep}: train loss {np.mean(losses):.4f} | dev token macro-F1 {dev_score:.4f}")
            if dev_score > best:
                best, bad = dev_score, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    log(f"early stop (best dev {best:.4f})")
                    break
        self.model.load_state_dict(best_state)
        self.best_dev = best
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict(self, token_lists: list[list[str]], batch_size: int = 128) -> list[list[str]]:
        self.model.eval()
        pieces, owners = [], []
        for di, toks in enumerate(token_lists):
            for s, e in self._windows(toks):
                pieces.append(self._encode(toks[s:e], None))
                owners.append((di, s, e))
        tags_per_doc = [["O"] * len(t) for t in token_lists]
        k = 0
        for batch in self._batches(pieces, batch_size, shuffle=False):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(**{n: v.to(self.device) for n, v in batch.items()}).logits
            pred = logits.argmax(-1).cpu().numpy()
            for row in pred:
                di, s, e = owners[k]
                enc = pieces[k]
                prev = None
                for pos, wid in enumerate(enc.word_ids()):
                    if wid is not None and wid != prev:
                        tags_per_doc[di][s + wid] = BIO[row[pos]]
                    prev = wid
                k += 1
        return [from_bio(t) for t in tags_per_doc]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: Path, **kw) -> "BertSegmenter":
        return cls(model_name=str(path), **kw)
