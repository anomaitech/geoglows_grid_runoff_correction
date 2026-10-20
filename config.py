"""Path configuration for the self-contained grid_runoff_correction package.

This is a local copy so the package does not import from the repo's src/.
`PROJECT_DIR` is where the DATA and RESULTS trees live. It defaults to the repo
root (two levels up from this file, i.e. .../ssro_store_experiment) so the bundled
trainer finds data/big/ and writes results/latpred_<tag>/. If you move this folder
away from the repo (or the data lives elsewhere), set the PROJECT_DIR env var:

    PROJECT_DIR=/path/to/ssro_store_experiment python train_lstm.py

NOTE: the package is code-isolated, but TRAINING still requires the data tree
(data/big/{era5,pairs,gagesii,catalog.csv}, + mswep/ for the champion) and PyTorch.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR",
                                  str(Path(__file__).resolve().parents[1])))
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
FIGURES = PROJECT_DIR / "figures"
RESULTS = PROJECT_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GAUGE_ID = "02177000"
GAUGE_LAT = 34.8139
GAUGE_LON = -83.3064

FORCING_START = "1980-01-01"
STUDY_START = "1981-01-01"
STUDY_END = "2023-12-31"
CAL_START, CAL_END = "1981-01-01", "2005-12-31"
EVAL_START, EVAL_END = "2006-01-01", "2023-12-31"
CFS_TO_CMS = 0.3048 ** 3
