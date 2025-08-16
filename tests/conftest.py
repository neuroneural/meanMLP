# tests/conftest.py
import os
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repo src/ is importable during tests (run pytest from repo root)
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Make MPS behave if present; harmless elsewhere
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

@pytest.fixture(scope="session")
def toy_data():
    """
    Deterministic, balanced binary dataset.
    We *know* where each class lives:
      - y[0:Z]   -> class 0
      - y[Z:B]   -> class 1
    """
    rng = np.random.default_rng(0)
    B, T, D = 24, 30, 10
    Z = B // 2  # zeros first, ones second

    y = np.zeros(B, dtype=np.int64)
    y[Z:] = 1

    X = rng.normal(size=(B, T, D)).astype("float32")
    # Inject signal into class 1
    X[Z:, :, :5] += 0.4
    X[Z:, 10:24, :] -= 0.2

    return X, y  # numpy arrays

@pytest.fixture(scope="session")
def dims(toy_data):
    X, y = toy_data
    _, _, D = X.shape
    C = int(np.unique(y).size)
    return D, C

@pytest.fixture(scope="session")
def toy_splits(toy_data):
    """
    Provide deterministic, balanced indices for train/val/test,
    using the known label layout from toy_data().
    Layout:
      zeros = [0:Z), ones = [Z:B)
    We take:
      train: 6 zeros + 6 ones
      val:   3 zeros + 3 ones
      test:  4 zeros + 4 ones
    (total uses 12+6+8 = 26 indices, but B=24; adjust to fit cleanly)
    """
    X, y = toy_data
    B = y.shape[0]
    Z = B // 2  # split point

    zeros = np.arange(0, Z)
    ones  = np.arange(Z, B)

    # choose counts (must sum <= class counts)
    n_tr, n_va, n_te = 6, 3, 3  # per-class counts

    tr_idx = np.concatenate([zeros[:n_tr], ones[:n_tr]])
    va_idx = np.concatenate([zeros[n_tr:n_tr+n_va], ones[n_tr:n_tr+n_va]])
    te_idx = np.concatenate([zeros[n_tr+n_va:n_tr+n_va+n_te],
                              ones[n_tr+n_va:n_tr+n_va+n_te]])

    # shuffle inside each split to mix classes a bit (deterministic)
    rng = np.random.default_rng(1)
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)

    splits = {
        "train": tr_idx,
        "val": va_idx,
        "test": te_idx,
    }
    return splits
