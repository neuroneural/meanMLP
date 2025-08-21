# tests/test_models_api.py
import importlib
import pandas as pd
import torch

def _first_batch(dataloader):
    for b in dataloader:
        return b
    return None

def test_exported_models_api(toy_data, dims, toy_splits):
    X, y = toy_data
    D, C = dims
    idx = toy_splits

    mdl = importlib.import_module("ml4fmri.models")
    names = list(getattr(mdl, "__all__", []))
    assert names, "ml4fmri.models.__all__ is empty; export your model classes there"

    # Models that don't follow standard PyTorch API (e.g., sklearn-based)
    excluded_models = ["LR"]

    failures = []

    for name in names:
        Model = getattr(mdl, name)

        def fail(msg):
            failures.append(f"[{name}] {msg}")

        # Skip excluded models entirely
        if name in excluded_models:
            continue

        # 1) constructor: (input_size, output_size)
        try:
            m = Model(D, C)
            if not isinstance(m, torch.nn.Module):
                fail("constructor did not produce a torch.nn.Module")
        except Exception as e:
            fail(f"constructor raised: {e}")
            continue  # skip the rest for this model

        # 2) dataloaders via class helper 
        if not hasattr(Model, "prepare_dataloader"):
            fail("missing class method prepare_dataloader")
            continue
        try:
            Xtr, ytr = X[idx["train"]], y[idx["train"]]
            Xva, yva = X[idx["val"]],   y[idx["val"]]
            Xte, yte = X[idx["test"]],  y[idx["test"]]

            train_dl = Model.prepare_dataloader(Xtr, ytr, shuffle=True)
            val_dl   = Model.prepare_dataloader(Xva, yva, shuffle=False)
            test_dl  = Model.prepare_dataloader(Xte, yte, shuffle=False)
        except Exception as e:
            fail(f"prepare_dataloader raised: {e}")
            continue

        # 3) one batch
        try:
            batch = _first_batch(train_dl)
            if batch is None:
                fail("train dataloader yielded no batches")
                continue

            bx, by = batch[:-1], batch[-1] # most of the time batch contains (data, labels), but sometimes there's more
            if not isinstance(by, torch.Tensor):
                fail("batch elements are not torch.Tensors")
        except Exception as e:
            fail(f"consuming first batch raised: {e}")
            continue

        # 4) forward returns (logits, dict)
        try:
            logits, loss_load = m(*bx)
            if not (isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[1] == C):
                fail(f"forward: logits has wrong shape {tuple(getattr(logits,'shape',()))}; expected (*, {C})")
            if not isinstance(loss_load, dict):
                fail("forward: second return is not a dict (loss_load)")
        except Exception as e:
            fail(f"forward raised: {e}")
            continue

        # 5) compute_loss works
        if not hasattr(m, "compute_loss"):
            fail("missing compute_loss")
        else:
            try:
                loss, loss_log = m.compute_loss(loss_load, by)
                if not (isinstance(loss, torch.Tensor) and torch.isfinite(loss).item()):
                    fail("compute_loss: loss is not finite")
                if not isinstance(loss_log, dict):
                    fail("compute_loss: logs is not a dict")
            except Exception as e:
                fail(f"compute_loss raised: {e}")

        # 6) handle_batch works
        if not hasattr(m, "handle_batch"):
            fail("missing handle_batch")
        else:
            try:
                loss2, blog = m.handle_batch(batch)
                if not (isinstance(loss2, torch.Tensor) and torch.isfinite(loss2).item()):
                    fail("handle_batch: loss is not finite")
                if not isinstance(blog, dict):
                    fail("handle_batch: blog is not a dict")
            except Exception as e:
                fail(f"handle_batch raised: {e}")

        # 7) get_optimizer (optional)
        if hasattr(m, "get_optimizer"):
            try:
                opt = m.get_optimizer(lr=getattr(m, "lr", 1e-3))
                import torch.optim as optim
                if not isinstance(opt, optim.Optimizer):
                    fail("get_optimizer did not return a torch.optim.Optimizer")
            except Exception as e:
                fail(f"get_optimizer raised: {e}")

        # 8) train_model (tiny run)
        if not hasattr(m, "train_model"):
            fail("missing train_model")
        else:
            try:
                tdf, sdf = m.train_model(train_dl, val_dl, test_dl, epochs=2, patience=1)
                if not (isinstance(tdf, pd.DataFrame) and "epoch" in tdf.columns):
                    fail("train_model: train_df is not a DataFrame with 'epoch'")
                if not isinstance(sdf, pd.DataFrame):
                    fail("train_model: test_df is not a DataFrame")
            except Exception as e:
                fail(f"train_model raised: {e}")

    # final summary
    assert not failures, "Model API failures:\n  - " + "\n  - ".join(failures)
