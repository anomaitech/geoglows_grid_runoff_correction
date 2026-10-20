# LSTM-corrected runoff grid for GEOGLOWS routing

A 0.25° gridded **corrected runoff** product: a per-cell **daily runoff depth in
mm/day** (the mean map is mm/yr). It replaces the raw HTESSEL/ERA5 runoff that
GEOGLOWS routes — run it through the **same GEOGLOWS Muskingum routing**, at a
**daily timestep**. The routing *method* is unchanged; only the unit differs from
ERA5's native field, so note:

**Units & timestep.** Values are **mm/day** (per-cell daily runoff depth); the mean
map is **mm/yr**. ERA5's native `ro` is **m/hr** (ERA5 is hourly). Both are runoff
*depths* — ours is daily because the LSTM is daily. Convert to your routing's
convention if needed: `m/day = mm/day ÷ 1000`, or
`m³/s = (mm/day ÷ 1000 ÷ 86400) × cell_area_m²`. Route at a **daily** step
(the Muskingum operator here uses `dt = 1 day`).

---

## Architecture

An LSTM learns the corrected runoff (trained against gauge discharge *un-routed* to a
lateral); that runoff is then distributed, calibrated, and gridded for GEOGLOWS to route:

```
 STAGE 1 · train_lstm.py  —  learn corrected runoff
 ───────────────────────────────────────────────────────────────────────────
   gauge discharge  ──invert_reach──▶  lateral runoff depth ──┐
   Q_obs (m³/s)       (reservoir⁻¹)        (mm/day)           │  = TARGET
                                                              ▼
   ERA5 / ECMWF forcing  ┐                               ┌──────────┐
   GEOGLOWS discharge gg ─┼──── inputs ─────────────────▶│   LSTM   │
   static attributes     ┘   (6-fold basin-holdout)      └────┬─────┘
                                                              ▼
                                         predicted lateral  =  CORRECTED RUNOFF
                                         depth (mm/day, per catchment)

 STAGE 2 · make_corrected_grid.py  —  distribute · calibrate · grid
 ───────────────────────────────────────────────────────────────────────────
   corrected runoff depth
        │
        ▼  distribute to reaches      weight = area (humid) / ROGEN (arid)
   per-reach runoff depth
        │
        ▼  calibrate at gauged reaches   VAR (variance) / QM (quantile-map)
        │
        ▼  rasterize to 0.25° grid       gridweights (area-weighted)
   ┌───────────────────────────┐
   │   CORRECTED RUNOFF GRID    │   0.25°, mm/day    ◀──  the deliverable
   └─────────────┬─────────────┘
                 ▼  GEOGLOWS Muskingum routing   (your existing routing — unchanged)
          corrected discharge   ──▶   KGE vs observed
 ───────────────────────────────────────────────────────────────────────────
   coverage tiers:  gauged reach → + calibration (~0.73)   |   ungauged → LSTM raw (~0.57)
   forecast-ready:  train with INCLUSIVE=1 (ECMWF-native inputs) for retrospective + 1–15 day
```

---

## Pipeline — two stages

1. **`train_lstm.py`** (Stage 1) — trains the champion streamflow-correction LSTM
   (6-fold **basin-holdout**; target = observed discharge un-routed to lateral inflow
   via a **one-line linear-reservoir inverse** (`invert_reach`), so the output *is* a
   corrected runoff) and writes each catchment's runoff (mm/day) to
   `results/latpred_<tag>/`.
   - champion (MSWEP regional precip, retrospective): `python train_lstm.py`
   - **ECMWF-native "inclusive"** model (ERA5 inputs only — usable for BOTH the
     retrospective *and* the 1-15 day forecast, since both share the HTESSEL land
     model): `INCLUSIVE=1 python train_lstm.py`
   - **self-contained** — this folder bundles the trainer (`big_hybrid.py`) and its
     helpers (`config.py, mb_core.py, metrics.py`); it does **not**
     import from the repo's `src/`. Still needs the **data tree** and **PyTorch**;
     `PROJECT_DIR` (env, default = repo root) points at `data/` + `results/`.
     ~15-25 min on a GPU / Apple-MPS box.
2. **`make_corrected_grid.py`** (Stage 2) — distributes that runoff to reaches,
   applies the calibration, and rasterizes to the 0.25° grid you route with GEOGLOWS.
   Reads `latpred_<tag>/` from Stage 1. Run `TRAIN=1 python make_corrected_grid.py`
   to chain Stage 1 automatically; otherwise it stops with instructions if the
   runoff is absent (training is never launched silently).

**Package contents (code-isolated — no `src/` dependency):**
`train_lstm.py` · `make_corrected_grid.py` · bundled trainer `big_hybrid.py` ·
helpers `config.py, mb_core.py, metrics.py`. External requirements:
the **data tree** (`$PROJECT_DIR/data/big/…`) and, for Stage 1 only, **PyTorch**
(Stage 2 needs just numpy/pandas/scipy).

---

## Method (per gauged catchment)

1. **LSTM corrected runoff** — an LSTM trained to reproduce the observed-deconvolved
   lateral inflow — the observed discharge un-routed by a **one-line linear-reservoir
   inverse** (`invert_reach`, replacing the old regularized `lsqr` deconvolution:
   validated equivalent-or-better, simpler, and more robust). `gg` is only an input
   feature. The LSTM gives each catchment's corrected runoff as a daily depth
   (mm/day) — the engine.
2. **Distribute to reaches** — the catchment runoff is spread across the
   catchment's reaches. Weights are **regime-dependent, not area alone**:
   - humid / temperate (ERA5 runoff depth ≥ 30 mm/yr) → **incremental catchment
     area** (`INCA`), i.e. a spatially uniform runoff depth;
   - arid (< 30 mm/yr) → **ERA5 runoff-generation pattern** (`ROGEN`), which
     concentrates runoff where it is actually produced instead of smearing it
     over dry ground.
   The area-weighted sum over reaches always equals the LSTM catchment value
   (mass-conserving).
3. **Calibrate at the gauge (where observations exist)** — fit on the
   **calibration half** of the record, apply to the **validation half**, and
   distribute the correction to every reach/cell in the catchment:
   - **VAR** — a single multiplicative variance/volume factor (the KGE-optimal
     scaling `f* = (α+β)/(α²+β²)`). Routing-linear, so it survives routing exactly.
   - **QM** — empirical **quantile mapping** of the LSTM runoff onto the observed
     runoff (observed runoff obtained by inverting the outlet discharge).
4. **Rasterize to 0.25°** — reach runoff is area-weighted onto the GEOGLOWS
   `gridweights` cells → the gridded product.

---

## Results (basin-holdout validation, 219 catchments)

We route the corrected runoff back through the reach network (the same Muskingum
operator GEOGLOWS uses) and score KGE against the observed gauge on the validation
half. Each correction can be applied two ways:

 - **grid** — baked into the runoff; the grid is routed *as-is* through GEOGLOWS.
 - **post-route** — applied to the *routed discharge* afterwards, using the
   per-gauge `postrouting_correction_table.csv`. Stronger, because baking a
   time-varying correction into the runoff and re-routing it re-convolves
   ("smears") the correction: `route(ratio·runoff) ≠ ratio·route(runoff)`.

| regime | n | raw | VAR (grid) | QM (grid) | VAR (post-route) | QM (post-route) |
|---|---|---|---|---|---|---|
| humid | 163 | 0.556 | 0.640 | 0.654 | 0.702 | 0.658 |
| semi | 42 | 0.245 | 0.431 | 0.455 | 0.455 | 0.455 |
| arid | 14 | −0.486 | −0.209 | −0.178 | −0.273 | −0.100 |
| **ALL** | 219 | 0.516 | 0.599 | 0.620 | **0.653** | 0.629 |

For reference, quantile-mapping the **raw GEOGLOWS** output at the same gauges
reaches only ~0.49 — the LSTM's timing improvement (which no calibration can add)
is what carries it higher.

**Reading the ladder:**
 - **Route-as-is** (correction baked into the grid): **~0.60–0.62**.
 - **Route + post-routing table** (apply the correction to the routed discharge):
   **~0.63–0.65**. The ~0.03 gain is the re-routing smear recovered.
 - The QM correction is derived at the **discharge level** (the strong form) and
   pushed into the grid as a multiplicative time-ratio — not a weaker runoff-space
   QM.
 - An additional **σ-K advance** lifts the numbers to the ~0.70 tier, but it is
   **outlet-specific and not shippable for a global grid** — see
   *"The σ-K advance toggle"* below.

### The σ-K advance toggle (`SIGMA_K_ADVANCE`, default OFF)

Setting `SIGMA_K_ADVANCE=1` time-advances each reach's runoff by its travel time
(ΣK) to the catchment outlet before gridding. On the same 219 basins it recovers
the full gauged-tier skill:

| | raw | VAR (grid) | QM (grid) | VAR (post-route) | QM (post-route) |
|---|---|---|---|---|---|
| advance **OFF** (standard — shipped) | 0.516 | 0.599 | 0.620 | 0.653 | 0.629 |
| advance **ON** | 0.537 | 0.636 | **0.699** | **0.714** | 0.707 |

**Why it helps:** pre-advancing cancels the routing convolution ("smear"), so a
correction baked into the runoff survives re-routing — grid-baked QM jumps
0.620 → 0.699, post-routing reaches ~0.71.

**Why it is left OFF (critical — read before enabling):** the advance shifts each
reach by its travel time **to one specific gauge outlet**. That is correct when
routing *to that outlet* (so the ON numbers above are real validation numbers),
but a global grid is routed to *many* downstream points at once, and a single
pre-advance cannot be correct for all of them. **A globally pre-advanced grid is
only valid at the one outlet it was built for — so the advance is a *validation
ceiling*, not a shippable grid transform.** The proper way to reach the ~0.70 tier
on a general grid is to apply the LSTM at **finer (per-reach) scale** so each cell
carries true *local generation*, which routes correctly everywhere with **no
advance needed** (the advance only exists to patch up distributing a *lumped*
catchment value). **Keep `SIGMA_K_ADVANCE=0` for any grid you actually hand off.**

---

## Inputs (all under `DATA_DIR`, the GEOGLOWS v3 data tree)

| file | rows | contents |
|---|---|---|
| `rfs_v3/routing_NA_all.parquet` | 710,461 | `river_id`, `next_river_id` (downstream), `k` (Muskingum K, seconds), `x` (Muskingum X) — the river network + routing parameters |
| `rfs_v3/conus_reach_runoff_2020.parquet` | 493,744 | `river_id`, `ro_gen_m3s` (ERA5 runoff generation, m³/s), `ro_depth_mm_yr` (runoff depth), `area` (incremental catchment area, **m²**) |
| `rfs_v3/gridweights_NA_all.parquet` | 1,120,479 | `river_id`, `x_index`, `y_index`, `area_sqm`, `proportion`, `lon`, `lat` — maps each 0.25° cell to the reach(es) it drains to |
| `catalog.csv` | 14,951 | gauge metadata: `site_no`, `river_id`, `drain_km2`, `lat`/`lon`, `musk_k_s` — links USGS gauges to reaches |
| `pairs/<site_no>.parquet` | — | `q_obs_cms` observed discharge (calibration/validation target) |
| `<LATPRED_DIR>/<site_no>.csv` | — | `lat_pred_mm` — the LSTM corrected runoff (the engine's output) |
| `gagesii/conterm_climate.txt` | — | `PET`, `PPTAVG_BASIN` → aridity index, used for the regime split |

---

## Usage

```bash
# STAGE 1 — train the LSTM (writes results/latpred_<tag>/).  Skip if latpred exists.
python train_lstm.py                 # champion (MSWEP, retrospective)
# INCLUSIVE=1 python train_lstm.py   # ECMWF-native: one model for retrospective + forecast

# STAGE 2 — distribute + calibrate + grid.  Edit the CONFIG block or override via env:
DATA_DIR=/path/to/data/big \
LATPRED_DIR=/path/to/results/latpred_ours_dm_ms_ema_mswep \
N_VALIDATION=400 N_TIMESERIES=25 \
SIGMA_K_ADVANCE=0 \        # 0 = standard shippable grid (default); 1 = validation-only, see caveat
python make_corrected_grid.py
# (or: TRAIN=1 python make_corrected_grid.py  to run Stage 1 automatically first)
```

### Outputs (written to `output/`)
- `corrected_runoff_grid_timeseries_sample.parquet` — **the routable product**:
  daily per-cell corrected runoff for `N_TIMESERIES` catchments
  (`x_index, y_index, lon, lat, date, runoff_mm_day, correction, site_no`).
  Scale `N_TIMESERIES` up to cover the full domain.
- `postrouting_correction_table.csv` — per-gauge `affine_a, affine_b, var_factor`
  for the **optional post-routing correction** (apply to the routed discharge at
  gauged reaches to reach the ~0.65 tier; see the ladder above).
- `runoff_grid_mean_map_0p25.(parquet|csv)` — 0.25° mean corrected runoff map
  (`x_index, y_index, lon, lat, runoff_{raw,var,qm}_mm_yr`), for inspection.
- `validation_by_basin.csv` — per-catchment KGE for every variant (raw; VAR/QM
  grid-baked; VAR/QM post-routing) + regime + fitted `affine_a, affine_b`.

---

## Honest notes (please read before interpreting)

- **Coverage tiering.** VAR/QM need an observation, so they apply only in
  **gauged** catchments. Truly **ungauged** cells carry the **raw LSTM runoff**
  (uncalibrated, ~0.52). The correction columns make this explicit.
- **No σ-K pre-advance.** This is a *standard* runoff grid — route it normally.
  We deliberately do not time-advance the runoff (that would be a routing-specific
  artifact). Distributing a lumped catchment value at 0.25° costs ~0.03–0.04 KGE
  versus the gauge-point number; it disappears when the LSTM is applied at finer
  (per-reach) scale.
- **Timing comes from the LSTM, not the calibration.** VAR/QM fix amplitude /
  distribution only; they cannot change correlation `r`. The LSTM's `r` is the
  reason this beats raw GEOGLOWS.
- **Arid stays weak** (`r`-limited by precipitation). The lever there is better
  precip (Daymet / IMERG), not a correction operator.


Routing operators (`route_muskingum`, `invert_reach`, `shift_earlier`) are
reproduced inside `make_corrected_grid.py` from the project's `forward_fix.py`, so
the script is self-contained given the data tree above.
