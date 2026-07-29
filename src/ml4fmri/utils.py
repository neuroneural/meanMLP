"""
Shared types and helpers.
"""

import random
import warnings

import numpy as np
import torch


# -----------------------------
# Reproducibility helpers
# -----------------------------

def _seed_everything(seed: int):
    """
    Seed the global RNGs (python, numpy, torch: CPU, CUDA, MPS).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds both the CPU and (if present) all CUDA generators
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # MPS has its own generator; not covered above


# -----------------------------
# Checkpointing helpers
# -----------------------------

def _save_checkpoint(model, path_no_ext):
    """
    Persist a trained model: torch as a state_dict (`.pt`), anything else via joblib.

    BasicTrainer restores best-validation weights before testing, so the live model
    already holds what we want. Failures only warn -- losing weights should not cost
    a whole run. Returns the written path, or None.
    """
    try:
        if isinstance(model, torch.nn.Module):
            path = path_no_ext + ".pt"
            torch.save(model.state_dict(), path)
        else:
            import joblib  # ships with scikit-learn
            path = path_no_ext + ".joblib"
            joblib.dump(model, path)
        return path
    except Exception as exc:  # never let checkpointing kill a run
        warnings.warn(
            f"Could not save checkpoint to {path_no_ext}: {exc}. "
            "Training results are unaffected.",
            RuntimeWarning,
        )
        return None
