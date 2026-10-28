"""Fine-tune the bi-encoder and the cross-encoder (own PyTorch loops, no `accelerate`).

Bi-encoder (sentence-transformers all-MiniLM-L6-v2): encodes query and report
separately into 384-d vectors; similarity = cosine. Trained with Multiple
Negatives Ranking loss: for a batch of (anchor, positive, hard negative) triples,
each anchor must score its own positive higher than every other positive and
every hard negative in the batch (cross-entropy over a similarity matrix x 20).
  positive       same family + same JASC code + same part condition
  hard negative  same family + same ATA chapter, different JASC code
                 (looks similar on the surface, but a different subsystem)
Batches never contain two anchors from the same (family, ata4, condition) group,
otherwise a true match would be treated as a negative.

Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2): reads query and candidate
TOGETHER ("[CLS] query [SEP] candidate"), so attention can compare them word by
word. Far more accurate than the bi-encoder but too slow to scan 400k reports, so
it only re-ranks the top 50. Trained with binary cross-entropy on bi-encoder
candidates of train queries (label 1 = proxy-relevant).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from aerofix.utils import get_logger

log = get_logger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mine_triples(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = train[train.problem_masked.str.len() > 0]
    key = train.family + "|" + train.ata4 + "|" + train.condition
    groups = train.groupby(key).indices
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    by_chapter = train.groupby(train.family + "|" + train.ata2).indices
    keys = list(groups)
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    p = np.sqrt(sizes) / np.sqrt(sizes).sum()  # sqrt sampling: big groups appear more, but don't dominate
    rows = []
    texts, ata4s = train.problem_masked.values, train.ata4.values
    for k in rng.choice(len(keys), size=n, p=p):
        a, b = rng.choice(groups[keys[k]], 2, replace=False)
        fam, a4, _ = keys[k].split("|")
        pool = by_chapter[f"{fam}|{a4[:2]}"]
        neg = None
        for _ in range(10):
            c = rng.choice(pool)
            if ata4s[c] != a4:
                neg = c
                break
        if neg is None:
            continue
        rows.append((keys[k], texts[a], texts[b], texts[neg]))
    return pd.DataFrame(rows, columns=["group", "anchor", "positive", "negative"])


def features(model, texts: list[str]) -> dict:
    """Tokenise for a SentenceTransformer. v6 renamed tokenize -> preprocess and adds a non-tensor
    'modality' entry, so only tensors are moved to the GPU."""
    prep = getattr(model, "preprocess", None) or model.tokenize
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in prep(texts).items()}


def unique_group_batches(groups: np.ndarray, batch_size: int, rng) -> list[list[int]]:
    order = rng.permutation(len(groups))
    batches, cur, seen = [], [], set()
    for i in order:
        if groups[i] in seen:
            continue
        cur.append(i)
        seen.add(groups[i])
        if len(cur) == batch_size:
            batches.append(cur)
            cur, seen = [], set()
    return batches  # leftovers (repeat groups) are simply dropped: a few % of triples


def train_biencoder(model, triples: pd.DataFrame, batch_size: int = 64, lr: float = 2e-5, scale: float = 20.0,
                    seed: int = 42):
    rng = np.random.default_rng(seed)
    batches = unique_group_batches(triples.group.values, batch_size, rng)
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(batches)), len(batches))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    def embed(texts):
        return F.normalize(model(features(model, [t.lower() for t in texts]))["sentence_embedding"], dim=-1)

    for step, b in enumerate(batches):
        t = triples.iloc[b]
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            a = embed(list(t.anchor))
            cands = embed(list(t.positive) + list(t.negative))
            sim = (a @ cands.T).float() * scale
            loss = F.cross_entropy(sim, torch.arange(len(b), device=DEVICE))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 200 == 0:
            log.info("bi-encoder step %d/%d loss %.4f", step, len(batches), loss.item())
    model.eval()
    return model


@torch.no_grad()
def encode(model, texts: list[str], batch_size: int = 512) -> np.ndarray:
    model.to(DEVICE).eval()
    texts = [t.lower() for t in texts]
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        feats = features(model, [texts[j] for j in idx])
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            e = model(feats)["sentence_embedding"]
        out[idx] = F.normalize(e.float(), dim=-1).cpu().numpy()
    return out


class CrossEncoder:
    def __init__(self, name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_len: int = 128):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(DEVICE)
        self.max_len = max_len

    def _batch(self, q, d):
        enc = self.tok([x.lower() for x in q], [x.lower() for x in d], truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        return {k: v.to(DEVICE) for k, v in enc.items()}

    def fit(self, queries, docs, labels, batch_size: int = 64, lr: float = 2e-5, seed: int = 42):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        steps = math.ceil(len(order) / batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
        pos_weight = torch.tensor([(1 - np.mean(labels)) / max(np.mean(labels), 1e-6)], device=DEVICE)
        self.model.train()
        for s in range(steps):
            b = order[s * batch_size:(s + 1) * batch_size]
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = self.model(**self._batch([queries[i] for i in b], [docs[i] for i in b])).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits.float(), torch.tensor(labels[b], dtype=torch.float32,
                                                          device=DEVICE), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if s % 500 == 0:
                log.info("cross-encoder step %d/%d loss %.4f", s, steps, loss.item())
        self.model.eval()
        return self

    @torch.no_grad()
    def score(self, query: str, docs: list[str], batch_size: int = 128) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(docs), batch_size):
            d = docs[i:i + batch_size]
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                out.append(self.model(**self._batch([query] * len(d), d)).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
