"""Classification metrics and plots.

macro-F1 is the headline: every class counts equally, so a model can't score well
by predicting ATA 53 (46% of train) all the time. Accuracy and top-3 accuracy are
reported next to it (top-3 matters for the app, which shows a ranked list).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def summary(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_F1": round(f1_score(y, pred, average="macro", labels=np.arange(probs.shape[1]), zero_division=0), 4),
            "weighted_F1": round(f1_score(y, pred, average="weighted", zero_division=0), 4),
            "top3_acc": round(float((top3 == y[:, None]).any(1).mean()), 4)}


def per_class(y: np.ndarray, probs: np.ndarray, names: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, probs.argmax(1), labels=np.arange(len(names)), zero_division=0)
    return pd.DataFrame({"class": names, "precision": p.round(3), "recall": r.round(3), "F1": f.round(3),
                         "support": s}).sort_values("support", ascending=False)


def plot_confusion(y: np.ndarray, probs: np.ndarray, names: list[str], title: str, path) -> None:
    """Row-normalised confusion matrix (each row = true class, sums to 1): readable despite imbalance."""
    cm = confusion_matrix(y, probs.argmax(1), labels=np.arange(len(names))).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 2, 0.45 * len(names) + 1.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "#0b0b0b")
    ax.set_xlabel("predicted", color="#52514e")
    ax.set_ylabel("true", color="#52514e")
    ax.set_title(title, loc="left", fontsize=10)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
