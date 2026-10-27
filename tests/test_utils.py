import random

import numpy as np
import pytest

from aerofix.utils import set_seed


def test_set_seed_repeats_python_and_numpy():
    set_seed(42)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(42)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_set_seed_repeats_torch():
    torch = pytest.importorskip("torch")
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    assert torch.equal(a, torch.rand(3))
