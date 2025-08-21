import importlib
import inspect
import pkgutil
import torch.nn as nn

def _is_model_class(obj):
    return (
        inspect.isclass(obj)
        and (issubclass(obj, nn.Module) or obj.__name__ == "LR")
        and hasattr(obj, "prepare_dataloader")
        and hasattr(obj, "train_model")
    )

def test_models_all_is_complete():
    # import/refresh the package
    mdl = importlib.import_module("ml4fmri.models")
    importlib.invalidate_caches()
    importlib.reload(mdl)

    # Collect exported names
    exported = set(getattr(mdl, "__all__", []))

    # Discover implemented classes by scanning submodules
    discovered = {}
    if hasattr(mdl, "__path__"):
        for info in pkgutil.iter_modules(mdl.__path__):
            modname = info.name
            if modname.startswith("_"):
                continue
            try:
                mod = importlib.import_module(f"{mdl.__name__}.{modname}")
            except Exception:
                continue

            # STRICT: class with same name as module
            cls = getattr(mod, modname, None)
            if _is_model_class(cls):
                discovered[cls.__name__] = cls
                continue

    # Assert every discovered model is exported in __all__
    assert set(discovered.keys()).issubset(exported), (
        "These implemented model classes are not exported in ml4fmri.models.__all__: "
        f"{sorted(set(discovered.keys()) - exported)}"
    )

    # Assert each exported name resolves to the actual class object
    for name in exported:
        obj = getattr(mdl, name, None)
        assert _is_model_class(obj), f"Exported name '{name}' is not a valid model class"

        # If discovered, ensure identity matches (same object)
        if name in discovered:
            assert obj is discovered[name], (
                f"Exported '{name}' does not match class found in its module"
            )
