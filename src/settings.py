"""Constants of the project"""
from datetime import datetime
import os
import platform

import path

PROJECT_ROOT = path.Path(os.path.dirname(__file__)).joinpath("..").abspath()
ASSETS_ROOT = PROJECT_ROOT.joinpath("assets")
LOGS_ROOT = ASSETS_ROOT.joinpath("logs")

UTCNOW = datetime.utcnow().strftime("%y%m%d.%H%M%S")

node = platform.node()
if "anon" in node:
    DATA_ROOT = path.Path("/anonpath/datasets")
else:
    DATA_ROOT = ASSETS_ROOT.joinpath("data")
