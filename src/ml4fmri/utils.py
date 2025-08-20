import torch
from importlib.resources import files, as_file

def load_packaged_weights(rel_path):
    """
    rel_path like 'weights/whole_milc.pth' inside ml4fmri.data
    Returns a loaded object/state_dict with torch.load.
    """
    res = files("ml4fmri.data").joinpath(rel_path)
    # as_file gives you a real path even when installed from a zip wheel
    with as_file(res) as p:
        return torch.load(p, map_location="cpu")