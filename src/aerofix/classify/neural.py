"""Shared multi-task training loop for the BiLSTM and transformer classifiers.

One model, three output heads (ata2, ata4, condition). The loss is the sum of the
three weighted cross-entropies, so the shared encoder learns features useful for
all targets (multi-task learning). Early stopping watches val ata2 macro-F1; the
best weights are kept in memory and restored at the end.

Length-grouped batching: texts are sorted by length inside random "mega-batches"
of 50 batches, then cut into batches and shuffled again. Each batch holds texts
of similar length, so padding (and wasted GPU work) is small, while batch order
stays random. Problem texts are short (median 21 sub-words) and a few are long,
so this makes epochs several times faster than padding everything to 128.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aerofix.classify.data import TARGETS
from aerofix.classify.metrics import summary


def length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(len(lengths))
    mega = batch_size * 50
    batches = []
    for s in range(0, len(idx), mega):
        chunk = idx[s:s + mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


@torch.no_grad()
def predict(model, seqs: list[list[int]], pad_id: int, device: str, batch_size: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    order = np.argsort([len(s) for s in seqs])  # sorted for speed, un-sorted at the end
    out = {t: [] for t in TARGETS}
    for i in range(0, len(order), batch_size):
        ids, mask = pad_batch([seqs[j] for j in order[i:i + batch_size]], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(ids.to(device), mask.to(device))
        for t in TARGETS:
            out[t].append(torch.softmax(logits[t].float(), -1).cpu().numpy())
    inv = np.argsort(order)
    return {t: np.concatenate(v)[inv] for t, v in out.items()}


def train(model, train_seqs, train_y: dict[str, np.ndarray], val_seqs, val_y: dict[str, np.ndarray],
          weights: dict[str, np.ndarray], *, pad_id: int, device: str, epochs: int, batch_size: int,
          make_optimizer, warmup_frac: float = 0.0, patience: int = 2, seed: int = 42, log=print,
          max_steps_per_epoch: int | None = None) -> tuple[torch.nn.Module, list[dict]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lengths = np.array([len(s) for s in train_seqs])
    steps_per_epoch = math.ceil(len(train_seqs) / batch_size)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    opt, sched = make_optimizer(model, steps_per_epoch * epochs, int(warmup_frac * steps_per_epoch * epochs))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    w = {t: torch.tensor(weights[t], dtype=torch.float32, device=device) for t in TARGETS}
    ys = {t: torch.tensor(train_y[t]) for t in TARGETS}

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for step, b in enumerate(length_grouped_batches(lengths, batch_size, rng)):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            ids, mask = pad_batch([train_seqs[i] for i in b], pad_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids.to(device), mask.to(device))
                loss = sum(F.cross_entropy(logits[t].float(), ys[t][b].to(device), weight=w[t]) for t in TARGETS)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            losses.append(loss.item())
            if step % 2000 == 0:
                log(f"  epoch {ep} step {step}/{steps_per_epoch} loss {np.mean(losses[-200:]):.4f}")
        probs = predict(model, val_seqs, pad_id, device)
        scores = {t: summary(val_y[t], probs[t]) for t in TARGETS}
        rec = {"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "minutes": round((time.perf_counter() - t0) / 60, 1),
               **{f"val_{t}_macro_F1": scores[t]["macro_F1"] for t in TARGETS},
               "val_ata2_acc": scores["ata2"]["accuracy"]}
        history.append(rec)
        log(json.dumps(rec))
        if rec["val_ata2_macro_F1"] > best:
            best, bad = rec["val_ata2_macro_F1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop: best val ata2 macro-F1 {best:.4f}")
                break
    model.load_state_dict(best_state)
    return model, history


def save_run(out_dir: Path, model, meta: dict, history: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
