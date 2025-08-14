# pylint: disable=line-too-long
""" A script for downloading a few fMRI datasets that were uploaded to github. 
    It will download a variant of ABIDE1 dataset (it is smaller than the one we used in our work)\
    and COBRE dataset, both ready for experiments within this framework.
    To use them in experiments, set `dataset=abide` or `dataset=cobre` when running `scripts/run_experiments.py`. 
"""
import requests
import os

from pathlib import Path
DATA_ROOT = Path(os.path.dirname(os.path.dirname(__file__))).joinpath("assets/data")

items = [
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/Data/ABIDE1_AllData.h5",
        DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/labels_ABIDE1.csv",
        DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/Data/COBRE_AllData.h5",
        DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/labels_COBRE.csv",
        DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
    ),
]

for item in items:
    r = requests.get(item[0], allow_redirects=True, timeout=100)
    os.makedirs(os.path.dirname(item[1]), exist_ok=True)
    with open(item[1], "wb") as f:
        f.write(r.content)