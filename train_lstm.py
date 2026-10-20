#!/usr/bin/env python3
"""
================================================================================
 Stage 1 of the corrected-runoff pipeline: TRAIN the LSTM
================================================================================

Trains the champion streamflow-correction LSTM and writes its per-basin corrected
runoff (lateral inflow, mm/day) to  results/latpred_<tag>/<site_no>.csv  — which
Stage 2 (make_corrected_grid.py) distributes to reaches and rasterizes to the
0.25° grid.

THE MODEL (architecture + training are the validated champion in src/big_hybrid.py):
  - inputs: ERA5/ECMWF dynamic forcing — runoff (sro, ssro), soil water (sw1,sw2),
    t2m, PET, radiation, snow (melt, swe, snowfall), precip (+ 3/7/30-day rolls) —
    plus GEOGLOWS retrospective discharge channels, a regional-precip slot, and
    GAGES-II static attributes (climate/topo/soils/landcover + log area, lat, lon,
    log reach-K).
  - target: the OBSERVED-deconvolved lateral inflow (invert each gauge's discharge
    through its reach reservoir). So the LSTM's output IS a corrected runoff.
  - head: fast flow + a physics slow-store whose time-constant is learned from the
    basin's statics (ENH_EMA); multi-timescale volume loss (ENH_MS).
  - protocol: 6-fold grouped BASIN-HOLDOUT — each basin is predicted by a model
    that never saw it, so the saved runoff is an out-of-fold (ungauged-style)
    estimate, which is what makes the Stage-2 KGE an honest ungauged number.

CONFIG (env). Defaults reproduce the shipped champion
(latpred_ours_dm_ms_ema_mswep, MSWEP regional precip, retrospective).
  INCLUSIVE=1  -> ECMWF-native model (ERA5 inputs only, NO MSWEP), written to
                 latpred_ours_era5_ms_ema. This is the single model usable for BOTH
                 the GEOGLOWS retrospective (ERA5) AND the 1-15 day forecast (ECMWF
                 ensemble) — both share the HTESSEL land model, only the driving
                 atmosphere differs, so an ECMWF-native model transfers to the
                 forecast (MSWEP has no forecast equivalent, hence it is dropped).
  EPOCHS, HID, STEPS, SEED  -> passed through to the trainer.

SELF-CONTAINED: this folder bundles the trainer (big_hybrid.py) and its helpers
(config.py, mb_core.py, mb_invert.py, metrics.py) -- it does NOT import from the
repo's src/. It still needs the DATA tree (data/big/{era5,pairs,gagesii,
catalog.csv}, plus mswep/ for the champion) and PyTorch. PROJECT_DIR (env, default
= repo root two levels up from this file) tells it where data/ and results/ live.
One run is ~15-25 min (6 folds) on a GPU / Apple-MPS box.

Usage:
    python train_lstm.py                 # champion  -> results/latpred_ours_dm_ms_ema_mswep/
    INCLUSIVE=1 python train_lstm.py     # inclusive -> results/latpred_ours_era5_ms_ema/
    PROJECT_DIR=/path/to/repo python train_lstm.py   # if data/ lives elsewhere
================================================================================
"""
import os
import sys
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent             # grid_runoff_correction/
TRAINER = HERE / "big_hybrid.py"                    # bundled trainer (helpers also bundled here)
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(HERE.parent)))

# ----- champion configuration (reproduces the shipped latpred) ----------------
if os.environ.get("INCLUSIVE", "0") == "1":
    os.environ.setdefault("DAYMET", "0")           # ECMWF-native (ERA5) inputs only
else:
    os.environ.setdefault("DAYMET", "1")           # use the regional-precip slot ...
    os.environ.setdefault("DAYMET_DIR", "mswep")   # ... fed with MSWEP (retrospective)
os.environ.setdefault("ENH_MS", "1")               # multi-timescale (30-day mean) loss
os.environ.setdefault("ENH_EMA", "1")              # physics slow-store head
os.environ.setdefault("SAVE_LAT", "1")             # write latpred_<tag>/<sid>.csv (Stage-2 input)

if not TRAINER.exists():
    sys.exit(f"ERROR: bundled trainer not found at {TRAINER} (expected big_hybrid.py "
             "alongside this script).")

tag = "latpred_ours_era5_ms_ema" if os.environ.get("DAYMET") == "0" \
      else "latpred_ours_dm_ms_ema_mswep"
print(f"[train_lstm] trainer : {TRAINER}  (self-contained; helpers bundled in this folder)")
print(f"[train_lstm] config  : DAYMET={os.environ.get('DAYMET')} "
      f"DAYMET_DIR={os.environ.get('DAYMET_DIR', '-')} ENH_MS=1 ENH_EMA=1 SAVE_LAT=1")
print(f"[train_lstm] data/out: {PROJECT_DIR}/{{data,results}}  "
      f"-> results/{tag}/<site_no>.csv (Stage-2 reads this)")

sys.path.insert(0, str(HERE))                      # resolve the BUNDLED config/mb_core/mb_invert/metrics
sys.argv = [str(TRAINER)]
runpy.run_path(str(TRAINER), run_name="__main__")
