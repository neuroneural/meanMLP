"""Constants used the project. Mostly paths to data, logs and so on"""

import datetime
import os
import platform
from path import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))
ASSETS_ROOT = PROJECT_ROOT.joinpath("assets")
WEIGHTS_ROOT = ASSETS_ROOT.joinpath("model_weights")
LOGS_ROOT = ASSETS_ROOT.joinpath("logs")

UTCNOW = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d_%H%M%S")

node = platform.node()
if "arctrd" in node:
    DATA_ROOT = Path("/data/users2/ppopov1/datasets")
else:
    DATA_ROOT = ASSETS_ROOT.joinpath("data")

if __name__ == "__main__":
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("ASSETS_ROOT:  ", ASSETS_ROOT)
    print("WEIGHTS_ROOT: ", WEIGHTS_ROOT)
    print("LOGS_ROOT:    ", LOGS_ROOT)
    print("DATA_ROOT:    ", DATA_ROOT)
    print("UTCNOW:       ", UTCNOW)
