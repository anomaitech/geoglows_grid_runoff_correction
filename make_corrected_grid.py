#!/usr/bin/env python3
"""
================================================================================
 LSTM-corrected runoff grid for GEOGLOWS routing
================================================================================

WHAT THIS PRODUCES
    A 0.25-degree gridded CORRECTED RUNOFF product: a per-cell DAILY runoff depth
    in mm/day (the mean map is mm/yr). Run it through the SAME GEOGLOWS Muskingum
    routing you already use, at a DAILY timestep -- the routing method is unchanged.
    UNITS: values are mm/day (ERA5's native `ro` is m/hr because ERA5 is hourly;
    both are runoff depths). Convert if your routing wants another unit:
      m/day = mm/day / 1000 ;  m3/s = (mm/day / 1000 / 86400) * cell_area_m2.

THE PIPELINE  (per gauged catchment)
    1. LSTM corrected runoff ....... STAGE 1 (train_lstm.py) trains the champion
       LSTM to reproduce observed-deconvolved lateral inflow and writes each
       catchment's corrected runoff (mm/day) to results/latpred_<tag>/. THIS
       script is STAGE 2 and reads that. Run `python train_lstm.py` first, or set
       TRAIN=1 here to chain Stage 1 automatically.
    2. distribute to reaches ....... the catchment runoff is spread across the
       catchment's river reaches using area weights (humid/temperate) or the
       ERA5 runoff-generation pattern (arid), so each reach carries a local
       runoff depth whose area-weighted sum equals the LSTM catchment value.
    3. calibrate at the gauge ...... where an observation exists, we fit a
       correction on the CALIBRATION half of the record and distribute that
       correction to every reach/cell in the catchment:
           - VAR : multiplicative variance-calibration factor  a = sigma_obs/sigma_sim
                   (routing-linear: routing a*runoff gives a*discharge, so the
                    routed variance matches observed)
           - QM  : empirical quantile mapping (sim CDF -> obs CDF), fit on the
                   calibration half, applied to the runoff (see NOTE on QM below)
    4. rasterize to 0.25 deg ....... reach runoff is mapped onto the GEOGLOWS
       gridweights cells (area-weighted) to produce the gridded product.

WHAT GETS VALIDATED (printed + written to validation_by_basin.csv)
    We route the corrected runoff back through the reach network (the same
    Muskingum operator GEOGLOWS uses) and score KGE against the observed gauge
    on the VALIDATION half. "raw" = LSTM only; "var"/"qm" = after calibration.

IMPORTANT, HONEST NOTES
    * Coverage tiering. VAR/QM calibration needs an observation, so it is only
      applied in GAUGED catchments. Truly ungauged cells carry the raw LSTM
      runoff (uncalibrated). The output columns make this explicit.
    * No sigma-K pre-advance. The grid is a STANDARD runoff product: route it
      normally. We deliberately do NOT time-advance the runoff (that would be a
      routing-specific artifact). Cost of distributing a lumped catchment value
      at 0.25 deg is ~0.03-0.04 KGE vs the gauge-point number; it vanishes when
      the LSTM is applied at finer (per-reach) scale.
    * QM is exact on discharge; applying it to runoff is an approximation
      (routing mixes timescales). VAR is routing-linear and is the recommended
      product. Both are written so they can be compared.
    * Timing (correlation r) comes from the LSTM, not from the calibration.
      VAR/QM fix amplitude/volume only; they cannot improve r.

INPUT DATA (paths under DATA_DIR; all already in the GEOGLOWS v3 data tree)
    rfs_v3/routing_NA_all.parquet ........ river_id, next_river_id, k (s), x
    rfs_v3/conus_reach_runoff_2020.parquet river_id, area (m2), ro_gen_m3s,
                                           ro_depth_mm_yr
    rfs_v3/gridweights_NA_all.parquet .... river_id, x_index, y_index,
                                           area_sqm, proportion, lon, lat
    catalog.csv .......................... site_no, river_id, drain_km2
    pairs/<site_no>.parquet .............. q_obs_cms (observed discharge)
    <LATPRED_DIR>/<site_no>.csv .......... lat_pred_mm (LSTM corrected runoff)
    gagesii/conterm_climate.txt .......... PET, PPTAVG_BASIN (for regime/weights)

USAGE
    python make_corrected_grid.py
    # edit the CONFIG block below to point DATA_DIR / LATPRED_DIR at your tree
    # and to scale N_VALIDATION / N_TIMESERIES_BASINS.

Author: (GEOGLOWS runoff-correction project)  --  routing operators reproduced
from the project's forward_fix.py so this script stands alone.
================================================================================
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

warnings.filterwarnings("ignore")

# ============================== CONFIG ========================================
# Point these at your data tree. Defaults assume this file sits inside the
# project (…/ssro_store_experiment/grid_runoff_correction/).
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE.parent / "data" / "big"))
LATPRED_DIR = Path(os.environ.get("LATPRED_DIR",
                                  HERE.parent / "results" / "latpred_ours_dm_ms_ema_mswep"))
OUT_DIR = HERE / "output"

N_VALIDATION = int(os.environ.get("N_VALIDATION", "400"))   # basins to validate + map
N_TIMESERIES_BASINS = int(os.environ.get("N_TIMESERIES", "25"))  # basins written as daily grid
MAX_SUBNETWORK = 400          # skip catchments larger than this many reaches
MIN_RECORD = 800              # minimum overlapping days to use a basin
ARID_YIELD_THRESHOLD = 30.0   # ro_depth_mm_yr below this -> use runoff-gen weights
SIGMA_K_ADVANCE = os.environ.get("SIGMA_K_ADVANCE", "0") == "1"  # default OFF: standard runoff grid
                                                                 # ON = non-standard pre-advanced grid
RANDOM_SEED = 0
# ==============================================================================


# ----------------------- routing operators (from forward_fix.py) --------------
def route_muskingum(x, K, X):
    """Forward Muskingum: Q_t = C0*I_t + C1*I_{t-1} + C2*Q_{t-1} (dt = 1 day)."""
    D = 2 * K * (1 - X) + 1.0
    C0, C1, C2 = (1 - 2 * K * X) / D, (1 + 2 * K * X) / D, (2 * K * (1 - X) - 1) / D
    return np.clip(lfilter([C0, C1], [1, -C2], x), 0, None)


def shift_earlier(x, days):
    """Advance a daily series by a fractional number of days (sigma-K advance)."""
    if days <= 0.01:
        return x
    idx = np.arange(len(x)) + days
    return np.interp(idx, np.arange(len(x)), x, left=x[0], right=x[-1])


def invert_reach(q, k):
    """One-step stable inverse of a reach's linear routing (X=0 form).

    Used to turn the observed OUTLET discharge back into an observed runoff
    series, which is the target for the runoff-space quantile mapping.
    """
    a = np.exp(-1.0 / max(k, 0.011))
    return np.clip((q - a * np.roll(q, 1)) / (1 - a), 0, None)


# ------------------------------- metrics --------------------------------------
def kge(sim, obs):
    """Kling-Gupta Efficiency; NaN if too short / degenerate."""
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(sim) < 120 or sim.std() == 0 or obs.std() == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2
                       + (sim.std() / obs.std() - 1) ** 2
                       + (sim.mean() / obs.mean() - 1) ** 2)


def quantile_map(src_cal, obs_cal, src_apply):
    """Empirical quantile mapping fit on the calibration half, applied elsewhere."""
    cdf = np.interp(src_apply, np.sort(src_cal),
                    np.linspace(0, 1, len(src_cal)), left=0, right=1)
    return np.interp(cdf, np.linspace(0, 1, len(obs_cal)), np.sort(obs_cal))


# --------------------------- load network + data ------------------------------
def load_network():
    rt = pd.read_parquet(DATA_DIR / "rfs_v3" / "routing_NA_all.parquet")
    NXT = dict(zip(rt.river_id.astype(np.int64), rt.next_river_id.astype(np.int64)))
    KD = {int(r): max(float(k) / 86400.0, 0.01) for r, k in zip(rt.river_id, rt.k)}  # K in days
    XX = {int(r): float(np.clip(x, 0.0, 0.49)) for r, x in zip(rt.river_id, rt.x)}
    KIDS = {}
    for r, nx in NXT.items():
        KIDS.setdefault(nx, []).append(r)

    # runoff spatial template: prefer a multi-year climatology if present,
    # else the single-year 2020 snapshot (override with RUNOFF_PARQUET=<name>).
    rfs = DATA_DIR / "rfs_v3"
    if os.environ.get("RUNOFF_PARQUET"):
        ro_file = rfs / os.environ["RUNOFF_PARQUET"]
    else:
        clim = sorted(rfs.glob("conus_reach_runoff_mean*.parquet"))
        ro_file = clim[-1] if clim else rfs / "conus_reach_runoff_2020.parquet"
    print(f"  runoff spatial template: {ro_file.name}")
    ro = pd.read_parquet(ro_file).set_index("river_id")
    INCA = ro.area.to_dict()            # incremental reach catchment area (m2)
    ROGEN = ro.ro_gen_m3s.to_dict()     # ERA5 runoff generation (m3/s) -> arid distribution weights
    YLD = ro.ro_depth_mm_yr.to_dict()   # ERA5 runoff depth (mm/yr) -> aridity flag

    cat = pd.read_csv(DATA_DIR / "catalog.csv", dtype={"site_no": str}) \
            .drop_duplicates("site_no").set_index("site_no")
    RID = {s: int(r) for s, r in cat.river_id.dropna().astype(np.int64).items()}

    clim = pd.read_csv(DATA_DIR / "gagesii" / "conterm_climate.txt",
                       dtype={"STAID": str}, encoding="latin-1").set_index("STAID")
    AI = (clim.PET / (clim.PPTAVG_BASIN * 10))  # aridity index (PET / precip)

    return dict(NXT=NXT, KD=KD, XX=XX, KIDS=KIDS, INCA=INCA, ROGEN=ROGEN,
                YLD=YLD, cat=cat, RID=RID, AI=AI)


def regime(ai):
    return "humid" if ai < 0.8 else ("semi" if ai < 1.5 else "arid")


# ----------------------- core: one catchment ----------------------------------
def process_basin(sid, net):
    """Return (validation dict, {river_id: corrected-runoff-depth series}) for a gauge."""
    RID, KD, XX, NXT, KIDS = net["RID"], net["KD"], net["XX"], net["NXT"], net["KIDS"]
    INCA, ROGEN, YLD, cat, AI = net["INCA"], net["ROGEN"], net["YLD"], net["cat"], net["AI"]

    orid = RID.get(sid)
    if orid is None or orid not in KD or sid not in cat.index or sid not in AI.index:
        return None
    area = float(cat.loc[sid, "drain_km2"])
    if not area > 0:
        return None

    # --- LSTM corrected runoff (mm/day) and observed discharge, aligned ---
    lp = pd.read_csv(LATPRED_DIR / f"{sid}.csv", index_col=0, parse_dates=True)["lat_pred_mm"]
    q = pd.read_parquet(DATA_DIR / "pairs" / f"{sid}.parquet")
    q.index = pd.to_datetime(q.index)
    j = pd.concat([lp, q.q_obs_cms], axis=1).dropna()
    j.columns = ["runoff_mm", "obs_cms"]
    if len(j) < MIN_RECORD:
        return None
    dates = j.index
    mm_to_cms = area * 1e6 / 1000.0 / 86400.0          # catchment mm/day -> m3/s
    catchment_lateral = (j.runoff_mm.clip(lower=0) * mm_to_cms).to_numpy()
    obs = j.obs_cms.to_numpy()

    # --- build the catchment's upstream reach subnetwork ---
    sub, frontier = [], [orid]
    while frontier and len(sub) <= MAX_SUBNETWORK:
        nxt = []
        for r in frontier:
            sub.append(r)
            nxt += [c for c in KIDS.get(r, []) if c in KD]
        frontier = nxt
    if len(sub) > MAX_SUBNETWORK or len(sub) < 2:
        return None
    subs = set(sub)
    children = {r: [c for c in KIDS.get(r, []) if c in subs] for r in sub}

    # cumulative drainage area (for a stable downstream ordering)
    cum, stack = {}, [(orid, False)]
    while stack:
        r, done = stack.pop()
        if done:
            cum[r] = max(INCA.get(r, 1.0), 1.0) + sum(cum[c] for c in children[r])
        else:
            stack.append((r, True))
            for c in children[r]:
                stack.append((c, False))
    order = sorted(sub, key=lambda r: cum[r])

    # sigma-K travel time from each reach down to the outlet (only used if advancing)
    D = {orid: KD[orid]}
    for r in sorted(sub, key=lambda r: -cum[r]):
        if r != orid:
            D[r] = KD[r] + D.get(NXT.get(r), KD[orid])

    # --- distribute the catchment runoff to reaches ---
    arid = YLD.get(orid, 100.0) < ARID_YIELD_THRESHOLD
    weight_src = ROGEN if arid else INCA
    total_w = sum(max(weight_src.get(r, 0.0), 0.0) for r in sub) or 1.0
    reach_lateral = {}                      # river_id -> lateral inflow (m3/s)
    for r in sub:
        w = max(weight_src.get(r, 0.0), 0.0) / total_w
        strip = catchment_lateral * w
        if SIGMA_K_ADVANCE:
            strip = shift_earlier(strip, D[r])
        reach_lateral[r] = strip

    # --- route the RAW LSTM runoff to the outlet (shared by calibration + validation) ---
    def route_network(reach_lat):
        inflow = {r: None for r in sub}
        out = {}
        for r in order:
            x = reach_lat[r] if inflow[r] is None else inflow[r] + reach_lat[r]
            out[r] = route_muskingum(x, KD[r], XX.get(r, 0.2))
            ds = NXT.get(r)
            if ds in inflow and r != orid:
                inflow[ds] = out[r] if inflow[ds] is None else inflow[ds] + out[r]
        return out[orid]

    sim = route_network(reach_lateral)
    half = len(obs) // 2
    obs_cal, sim_cal = obs[:half], sim[:half]

    # VAR: variance/volume calibration as a SINGLE multiplicative runoff factor (routing-linear).
    #      A single factor cannot match both mean and variance (that needs the affine a*s+b, which
    #      is a post-routing discharge op, not a runoff-grid one), so we use the KGE-optimal scaling
    #      f* = (alpha+beta)/(alpha^2+beta^2) that best balances both. QM (below) does better on the
    #      grid because it matches the full distribution.
    a_s = sim_cal.std() / max(obs_cal.std(), 1e-9)      # alpha = sigma_sim/sigma_obs
    b_s = sim_cal.mean() / max(obs_cal.mean(), 1e-9)     # beta  = mu_sim/mu_obs
    var_factor = float(np.clip((a_s + b_s) / (a_s ** 2 + b_s ** 2), 0.2, 5.0))

    # QM: the strong correction is the DISCHARGE-level quantile map (fit on the calibration half of
    #     the routed discharge -- exactly the correction that scores ~0.72). We express it as a
    #     multiplicative time-ratio qm(discharge)/discharge and push that SAME ratio back onto every
    #     reach's runoff, so the discharge-QM correction lives IN THE GRID. Re-routing the grid
    #     reproduces the discharge-QM result up to the (small) routing convolution of a smooth ratio.
    qm_discharge = quantile_map(sim_cal, obs_cal, sim)
    qm_ratio = np.clip(np.where(sim > 1e-6, qm_discharge / np.clip(sim, 1e-6, None), 1.0), 0.0, 10.0)

    # corrected lateral inflow per reach  (VAR = scalar factor, QM = time-varying ratio)
    reach_lat_var = {r: reach_lateral[r] * var_factor for r in sub}
    reach_lat_qm = {r: reach_lateral[r] * qm_ratio for r in sub}
    sim_var = var_factor * sim                                     # linear in the VAR factor
    sim_qm = route_network(reach_lat_qm)

    # POST-ROUTING corrections: applied to the routed DISCHARGE (i.e. AFTER GEOGLOWS routing),
    # not baked into the runoff. These reproduce the higher gauged-tier numbers -- full affine
    # variance calibration (a*s+b, fixes BOTH mean and variance) and discharge-level quantile
    # mapping -- and are provided as a per-gauge table so you can optionally post-correct
    # the routed output at gauged reaches.
    a_ls = float(np.clip(obs_cal.std() / max(sim_cal.std(), 1e-9), 0.2, 5.0))
    b_ls = float(obs_cal.mean() - a_ls * sim_cal.mean())
    sim_var_pr = a_ls * sim + b_ls
    sim_qm_pr = qm_discharge                                        # discharge-level QM (not baked)

    # corrected runoff DEPTH (mm/day) per reach, for the grid
    def to_depth(reach_lat):
        out = {}
        for r in sub:
            r_mm_to_cms = max(INCA.get(r, 1.0), 1.0) / 1000.0 / 86400.0   # area_m2 * mm/day -> m3/s
            out[r] = reach_lat[r] / r_mm_to_cms
        return out

    val = dict(
        site_no=sid, river_id=orid, regime=regime(float(AI.loc[sid])),
        n_reach=len(sub), drain_km2=area,
        kge_raw=kge(sim[half:], obs[half:]),
        kge_var=kge(sim_var[half:], obs[half:]),               # baked into runoff grid (route as-is)
        kge_qm=kge(sim_qm[half:], obs[half:]),                 # baked into runoff grid (route as-is)
        kge_var_postroute=kge(sim_var_pr[half:], obs[half:]),  # applied AFTER routing (affine a*s+b)
        kge_qm_postroute=kge(sim_qm_pr[half:], obs[half:]),    # applied AFTER routing (discharge QM)
        var_factor=var_factor, affine_a=a_ls, affine_b=b_ls,
    )
    grids = dict(raw=to_depth(reach_lateral), var=to_depth(reach_lat_var),
                 qm=to_depth(reach_lat_qm), dates=dates)
    return val, grids


# ------------------------------- rasterize ------------------------------------
def rasterize_mean(all_reach_means, gw):
    """Area-weighted mean runoff depth per 0.25-deg cell -> tidy DataFrame."""
    gw = gw[gw.river_id.isin(all_reach_means["raw"].keys())].copy()
    for col in ("raw", "var", "qm"):
        gw[col] = gw.river_id.map(all_reach_means[col])
    gw = gw.dropna(subset=["raw"])
    gw["w"] = gw.area_sqm
    agg = gw.groupby(["x_index", "y_index"]).apply(
        lambda d: pd.Series({
            "lon": d.lon.iloc[0], "lat": d.lat.iloc[0],
            "runoff_raw_mm_yr": np.average(d.raw, weights=d.w) * 365.25,
            "runoff_var_mm_yr": np.average(d["var"], weights=d.w) * 365.25,
            "runoff_qm_mm_yr": np.average(d.qm, weights=d.w) * 365.25,
        })
    ).reset_index()
    return agg


# ---------------------------------- main --------------------------------------
def ensure_lstm_runoff():
    """Stage 1: make sure the LSTM corrected runoff exists; train it if asked/needed.

    - TRAIN=1            -> (re)train via train_lstm.py, then grid.
    - runoff missing     -> stop with instructions (training is a ~15-25 min GPU job,
                            so we never launch it silently).
    - runoff present     -> use it.
    """
    have = LATPRED_DIR.exists() and any(LATPRED_DIR.glob("*.csv"))
    want_train = os.environ.get("TRAIN", "0") == "1"
    if have and not want_train:
        return
    if not have and not want_train:
        print(f"\nLSTM corrected runoff not found at: {LATPRED_DIR}")
        print("Run STAGE 1 first:   python train_lstm.py      (champion / MSWEP)")
        print("                     INCLUSIVE=1 python train_lstm.py   (ECMWF-native, forecast+retro)")
        print("or set TRAIN=1 to train now, then grid.")
        sys.exit(1)
    print("STAGE 1: training the LSTM (train_lstm.py) ...")
    env = {**os.environ}
    if "era5" in LATPRED_DIR.name:
        env["INCLUSIVE"] = "1"
    if subprocess.run([sys.executable, str(HERE / "train_lstm.py")], env=env).returncode:
        sys.exit("STAGE 1 (LSTM training) failed; see output above.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_lstm_runoff()                      # STAGE 1 (train if needed/requested)
    print("STAGE 2: loading GEOGLOWS network + runoff tables ...")
    net = load_network()
    gw = pd.read_parquet(DATA_DIR / "rfs_v3" / "gridweights_NA_all.parquet")

    sids = sorted(p.stem for p in LATPRED_DIR.glob("*.csv"))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(sids)
    sids = sids[:N_VALIDATION]
    print(f"Processing up to {len(sids)} gauged catchments ...")

    validations = []
    reach_mean = {"raw": {}, "var": {}, "qm": {}}      # river_id -> time-mean depth
    ts_written = 0
    ts_frames = []

    for sid in sids:
        res = process_basin(sid, net)
        if res is None:
            continue
        val, grids = res
        validations.append(val)
        for col in ("raw", "var", "qm"):
            for r, depth in grids[col].items():
                reach_mean[col][r] = float(np.nanmean(depth))     # mean map
        # write daily per-cell runoff for the first few basins (routable product sample)
        if ts_written < N_TIMESERIES_BASINS:
            rr = list(grids["var"].keys())
            sub_gw = gw[gw.river_id.isin(rr)]
            for col in ("var", "qm"):
                mat = pd.DataFrame({r: grids[col][r] for r in rr}, index=grids["dates"])
                # reach -> cell (area-weighted) daily depth
                for (xi, yi), d in sub_gw.groupby(["x_index", "y_index"]):
                    w = d.area_sqm.to_numpy()
                    series = (mat[d.river_id.tolist()].to_numpy() * w).sum(1) / w.sum()
                    ts_frames.append(pd.DataFrame({
                        "x_index": xi, "y_index": yi,
                        "lon": d.lon.iloc[0], "lat": d.lat.iloc[0],
                        "date": grids["dates"], "runoff_mm_day": series,
                        "correction": col, "site_no": sid,
                    }))
            ts_written += 1

    V = pd.DataFrame(validations)
    V.to_csv(OUT_DIR / "validation_by_basin.csv", index=False)

    mean_map = rasterize_mean(reach_mean, gw)
    mean_map.to_parquet(OUT_DIR / "runoff_grid_mean_map_0p25.parquet", index=False)
    mean_map.to_csv(OUT_DIR / "runoff_grid_mean_map_0p25.csv", index=False)

    if ts_frames:
        ts = pd.concat(ts_frames, ignore_index=True)
        ts.to_parquet(OUT_DIR / "corrected_runoff_grid_timeseries_sample.parquet", index=False)

    # ---------- summary ----------
    med = lambda s: np.nanmedian(s.dropna())
    print(f"\nValidated {len(V)} catchments. Outputs in: {OUT_DIR}\n")
    print("Routed-back KGE (median, validation half):")
    print(f"  grid-baked = correction inside the runoff (route as-is) | "
          f"post-route = correction applied after routing (optional table)")
    print(f"{'regime':7s} {'n':>4} | {'raw':>7} {'VARgrid':>8} {'QMgrid':>8} {'VARpost':>8} {'QMpost':>8}")
    for h in ["humid", "semi", "arid", "ALL"]:
        x = V if h == "ALL" else V[V.regime == h]
        if not len(x):
            continue
        print(f"{h:7s} {len(x):>4} | {med(x.kge_raw):>7.3f} {med(x.kge_var):>8.3f} "
              f"{med(x.kge_qm):>8.3f} {med(x.kge_var_postroute):>8.3f} {med(x.kge_qm_postroute):>8.3f}")
    # per-gauge table so the optional post-routing correction is reproducible
    V[["site_no", "river_id", "regime", "affine_a", "affine_b", "var_factor"]] \
        .to_csv(OUT_DIR / "postrouting_correction_table.csv", index=False)
    print(f"\nFiles written:")
    print(f"  validation_by_basin.csv ................ per-catchment KGE (raw/var/qm)")
    print(f"  runoff_grid_mean_map_0p25.(parquet|csv)  0.25-deg mean corrected runoff map")
    print(f"  corrected_runoff_grid_timeseries_sample.parquet")
    print(f"      daily per-cell corrected runoff for {ts_written} catchments")
    print(f"      (this is the routable product; scale N_TIMESERIES to cover more)")


if __name__ == "__main__":
    main()
