"""Constants used the project. Mostly paths to data, logs and so on"""

import datetime
import os
import platform

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")
WEIGHTS_ROOT = os.path.join(ASSETS_ROOT, "model_weights")
LOGS_ROOT = os.path.join(ASSETS_ROOT, "logs")

UTCNOW = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d_%H%M%S")

node = platform.node()
if "arctrd" in node:
    DATA_ROOT = "/data/users2/ppopov1/datasets"
else:
    DATA_ROOT = os.path.join(ASSETS_ROOT, "data")

if __name__ == "__main__":
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("ASSETS_ROOT:  ", ASSETS_ROOT)
    print("WEIGHTS_ROOT: ", WEIGHTS_ROOT)
    print("LOGS_ROOT:    ", LOGS_ROOT)
    print("DATA_ROOT:    ", DATA_ROOT)
    print("UTCNOW:       ", UTCNOW)
