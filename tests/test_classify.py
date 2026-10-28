import numpy as np
import pandas as pd
import torch

from aerofix.classify.data import LabelSpace, class_weights
from aerofix.classify.metrics import summary
from aerofix.classify.neural import length_grouped_batches, pad_batch

CFG = {"classify": {"top_k_ata2": 2, "ata4_min_count": 2, "top_n_condition": 1, "condition_exclude": ["", "UNKNOWN"]}}


def _train():
    return pd.DataFrame({"ata2": ["53", "53", "53", "33", "33", "25"],
                         "ata4": ["5310", "5310", "5320", "3350", "3350", "2560"],
                         "condition": ["UNKNOWN", "UNKNOWN", "UNKNOWN", "CRACKED", "CRACKED", "CORRODED"]})


def test_label_space_top_k_other_and_placeholders():
    ls = LabelSpace.fit(_train(), CFG)
    assert ls.names("ata2") == ["33", "53", "OTHER"]
    assert ls.names("ata4") == ["3350", "5310", "OTHER"]
    assert ls.names("condition") == ["CRACKED", "OTHER"]  # UNKNOWN excluded even though it is most frequent
    assert ls.encode("ata2", pd.Series(["53", "25", "99"])).tolist() == [1, 2, 2]


def test_class_weights_rarer_class_weighs_more():
    w = class_weights(np.array([0, 0, 0, 0, 1]), 3, power=0.5)
    assert w[1] > w[0] and w[2] == 0.0  # unseen class gets weight 0
    assert np.isclose(w[:2].mean(), 1.0)


def test_summary_perfect_and_top3():
    y = np.array([0, 1, 2])
    probs = np.eye(3)
    s = summary(y, probs)
    assert s["accuracy"] == 1.0 and s["macro_F1"] == 1.0 and s["top3_acc"] == 1.0


def test_length_grouped_batches_cover_all_once():
    lengths = np.random.default_rng(0).integers(1, 100, size=1000)
    batches = length_grouped_batches(lengths, 32, np.random.default_rng(1))
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(1000))
    # within a batch lengths are close (sorted inside mega-batches)
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in batches])
    assert spread < 20


def test_pad_batch():
    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert ids.dtype == torch.long
