"""HYBRID at scale: full weather forcing + GEOGLOWS retro + statics(+k).

Pool = screened basins (GAGES-II attrs, dams < STOR_MAX, verified reach)
that already have assembled ERA5-Land forcing. Lateral supervision as in
the 0.70 model; inputs are the union of everything measured to matter:

  dynamic (26): ERA5-Land tp (+3/7/30d), sro, ssro, sw1, sw2, t2m, ev7,
                melt (+7d), swe, snowfall, pet, srad, liquid input
                (+7/30d), aridity index, season (sin/cos),
                GEOGLOWS retro Q (log; +3/7/30d rolls, 1d lag)
  static  (~20): GAGES-II climate/topo/soils/landcover + log_area,
                lat, lon, log_k (reach routing timescale)

Target: deconvolved observed lateral (cached from big_train).
Output: corrected lateral (mm/day) -> muskingum(k) at deployment.
Protocol: 6-fold grouped basin-holdout, val = 2nd half of record.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from config import PROJECT_DIR, RESULTS
from mb_core import reservoir, panel


def timing_volume(sim, obs, npeaks=12, maxlag=10):
    """Explicit timing and volume diagnostics beyond KGE."""
    m = sim.notna() & obs.notna()
    s_, o_ = sim[m].to_numpy(), obs[m].to_numpy()
    out = {"beta": float(s_.mean() / max(o_.mean(), 1e-9)),
           "gamma": float((s_.std() / max(s_.mean(), 1e-9))
                          / max(o_.std() / max(o_.mean(), 1e-9), 1e-9))}
    sa, oa = s_ - s_.mean(), o_ - o_.mean()
    lags = range(-maxlag, maxlag + 1)
    xc = [np.corrcoef(sa[max(0, -l):len(sa) - max(0, l)],
                      oa[max(0, l):len(oa) - max(0, -l)])[0, 1] for l in lags]
    out["xlag_d"] = float(list(lags)[int(np.nanargmax(xc))])
    idx = np.argsort(o_)[-npeaks * 30:]
    peaks, used = [], np.zeros(len(o_), bool)
    for i in idx[::-1]:
        if used[max(0, i - 5):i + 6].any():
            continue
        used[i] = True
        peaks.append(i)
        if len(peaks) >= npeaks:
            break
    plags = []
    for p in peaks:
        w = slice(max(0, p - 7), min(len(s_), p + 8))
        plags.append(int(np.argmax(s_[w])) + w.start - p)
    out["peak_lag_d"] = float(np.median(plags)) if plags else np.nan
    out["abs_peak_lag_d"] = float(np.median(np.abs(plags))) if plags else np.nan
    return out
from mb_invert import kernel, conv_matrix, invert

BIG = PROJECT_DIR / "data" / "big"
GII = BIG / "gagesii"
SEED = int(os.environ.get("LSTM_SEED", "21"))
RNG = np.random.default_rng(21)
TRNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ = 365
EPOCHS = int(os.environ.get("EPOCHS", "22"))
HID = int(os.environ.get("HID", "192"))
STOR_MAX = float(os.environ.get("STOR_MAX", "200"))
STEPS = int(os.environ.get("STEPS", "150"))
# DPL=1: differentiable-parameter-learning end-to-end DISCHARGE loss. Route the
# predicted lateral through a differentiable linear reservoir (basin K) INSIDE the
# forward pass and add a loss on the routed discharge vs observed (log, normalized).
# DPL_W scales it; DPL_LEARNK=1 lets the net predict a K-multiplier from statics.
DPL = os.environ.get("DPL", "0") == "1"
DPL_W = float(os.environ.get("DPL_W", "1.0"))
DPL_LEARNK = os.environ.get("DPL_LEARNK", "0") == "1"
# VAR_LOSS=1: variance-aware loss. Penalize the predicted lateral sequence for
# having LOWER variability than the target over the window (attacks the arid
# under-dispersion alpha=0.28 that caps KGE). VAR_W scales it.
VAR_LOSS = os.environ.get("VAR_LOSS", "0") == "1"
VAR_W = float(os.environ.get("VAR_W", "0.5"))
# VAR_CV=1: mean-preserving dispersion penalty (match CV=sigma/mu, not raw sigma) so
# raising variance can't inflate the mean -- fixes the arid beta->1.22 overshoot.
# TIME_LOSS=1: windowed Pearson-correlation loss that DIRECTLY attacks the KGE r
# (timing) term -- tests whether arid r is improvable by optimization. TIME_W scales it.
VAR_CV = os.environ.get("VAR_CV", "0") == "1"
TIME_LOSS = os.environ.get("TIME_LOSS", "0") == "1"
TIME_W = float(os.environ.get("TIME_W", "0.5"))
# VOL_ANCHOR=1: pin predicted window-mean to target window-mean (penalize (beta-1)^2 at
# the window level) so the timing/CV loss can't inflate volume (the arid beta->1.31
# overshoot that ate the r gain). VOL_W scales it.
VOL_ANCHOR = os.environ.get("VOL_ANCHOR", "0") == "1"
VOL_W = float(os.environ.get("VOL_W", "0.5"))
# GOOGLE_STYLE=1: replicate the Nearing-et-al architecture family on OUR
# data — direct discharge target (area-normalized), no routing constraint,
# hidden 256, asymmetric-Laplacian likelihood head (1-component CMAL).
GSTYLE = os.environ.get("GOOGLE_STYLE", "0") == "1"
# ARCH_ONLY=1: Google's ARCHITECTURE (hidden 256, asymmetric-Laplacian
# probabilistic head) but OUR input/output contract: lateral target,
# routed evaluation. Isolates pure network-design differences.
ARCH_ONLY = os.environ.get("ARCH_ONLY", "0") == "1"
# LIQUID=1: liquid network (CfC closed-form continuous-time) in place of
# the LSTM cell; our loss, our I/O — pure cell-architecture ablation.
LIQUID = os.environ.get("LIQUID", "0") == "1"
# DAYMET=1: add the regional-precipitation slot (measured +0.12 KGE);
# pool restricted to basins with fetched Daymet.
USE_DAYMET = os.environ.get("DAYMET", "0") == "1"
# ZONES=1: zoned inputs, one output — 3 travel-time zone rains (near/
# middle/far, data/big/zone_rain/) + 3d/7d rolls; pool restricted to
# basins with a zone file.
USE_ZONES = os.environ.get("ZONES", "0") == "1"
# IMERG=1: use satellite precip (data/big/imerg/) as the rain channel;
# pool restricted to basins with IMERG, record trimmed to >=2001 (sat era).
USE_IMERG = os.environ.get("IMERG", "0") == "1"
# SITES=<file>: restrict the pool to a listed subset (fast pilots).
SITES_FILE = os.environ.get("SITES", "")
# ERA5_DIR: alternate channel source dir (e.g. era5_om for the
# Open-Meteo-hybrid ablation) — same file format as era5/.
ERA5_DIR = os.environ.get("ERA5_DIR", "era5")
# DAYMET_DIR: alternate precipitation source for the Daymet slot
# (e.g. a PREDICTED corrected precipitation) — same file format.
DAYMET_DIR = os.environ.get("DAYMET_DIR", "daymet")
# result-name tag for alternate sources: daymet_synth -> synth,
# daymet_synth3 -> synth3 (keeps corrector versions from overwriting)
SYN = "" if DAYMET_DIR == "daymet" else DAYMET_DIR.replace("daymet_", "")
# NO_TP=1: exclude ERA5 precipitation from the inputs (tp + rolls, and its
# contribution to liq/aridx) — measures how much the model leans on ERA5
# rain given soil water, runoff fields, and the GEOGLOWS discharge channel.
NO_TP = os.environ.get("NO_TP", "0") == "1"
# SAVE_MODELS=1: persist each fold's trained net + normalization so the
# out-of-fold model can be applied to NEW dates (forecast mode).
SAVE_MODELS = os.environ.get("SAVE_MODELS", "0") == "1"
# CORR_RUNOFF=1: replace sro/ssro inputs with the 4-param component-corrected
# versions (obs-derived T/gain targets, IDW-transferred leave-self-out).
CORR_RUNOFF = os.environ.get("CORR_RUNOFF", "0") == "1"
if CORR_RUNOFF:
    _TP = pd.read_csv(RESULTS/"T_params_xfer.csv", dtype={"gid": str}).set_index("gid")
def _acf_T(x):
    x = pd.Series(x).dropna()
    if len(x) < 500 or x.std() == 0: return np.nan
    r1 = x.autocorr(1)
    if not np.isfinite(r1) or r1 <= 0.01 or r1 >= 0.999: return np.nan
    return -1.0/np.log(r1)
def _retime(x, Tm, To):
    if not (np.isfinite(Tm) and np.isfinite(To)): return x
    if Tm > To + 0.2:
        a = np.exp(-1/(Tm-To))
        y = (x - a*np.roll(x,1))/(1-a); y[0] = x[0]
        return np.clip(y, 0, None)
    elif To > Tm + 0.2:
        return reservoir(x, To - Tm)
    return x
# NO_GG=1: exclude the GEOGLOWS retrospective-discharge channels — measures
# the price of full independence from their simulation (standalone model).
NO_GG = os.environ.get("NO_GG", "0") == "1"
# CARAVAN_CH=1: restrict to channels reproducible from Caravan forcing
# (global-training feasibility): drop sro/ssro (HTESSEL runoff), ev7
# (actual ET), gg (implies NO_GG); melt := clip(-dSWE,0), snowfall := tp
# on freezing days. sw1/sw2/swe/pet/srad/tp/t2m all exist in Caravan.
CARAVAN_CH = os.environ.get("CARAVAN_CH", "0") == "1"
if CARAVAN_CH:
    NO_GG = True
# Architecture enhancements (ours):
#  ENH_MS=1  multi-timescale loss — match 30-day mean of output sequence
#            to 30-day mean of target (attacks volume bias directly)
#  ENH_EMA=1 physics-informed slow path — output = fast + store(slow),
#            store time-constant learned from the basin's statics
ENH_MS = os.environ.get("ENH_MS", "0") == "1"
ENH_EMA = os.environ.get("ENH_EMA", "0") == "1"
if GSTYLE or ARCH_ONLY:
    HID = int(os.environ.get("HID", "256"))


def gtab(name, cols):
    df = pd.read_csv(GII / name, dtype={"STAID": str}, encoding="latin-1")
    return df.set_index("STAID")[cols]


# ---- STATIC LSTM inputs: per-catchment attributes from GAGES-II -------------
ATT = pd.concat([
    # climate: mean annual precip, mean temp, potential ET, snow fraction, precip seasonality
    gtab("conterm_climate.txt", ["PPTAVG_BASIN", "T_AVG_BASIN", "PET",
                                 "SNOW_PCT_PRECIP", "PRECIP_SEAS_IND"]),
    # topography: mean basin elevation, mean slope
    gtab("conterm_topo.txt", ["ELEV_MEAN_M_BASIN", "SLOPE_PCT"]),
    # soils: permeability, available water capacity, clay %, sand %, depth-to-bedrock
    gtab("conterm_soils.txt", ["PERMAVE", "AWCAVE", "CLAYAVE", "SANDAVE",
                               "ROCKDEPAVE"]),
    # land cover (NLCD 2006): forest, planted/cultivated, developed (+ grass/shrub/barren)
    gtab("conterm_lc06_basin.txt", (["FORESTNLCD06", "PLANTNLCD06", "DEVNLCD06"]
        + (["GRASSNLCD06", "SHRUBNLCD06", "BARRENNLCD06"]
           if os.environ.get("EXTRA_LC", "1") == "1" else []))),
], axis=1)
cls = pd.read_csv(GII / "conterm_bas_classif.txt", dtype={"STAID": str},
                  encoding="latin-1").set_index("STAID")
dams = pd.read_csv(GII / "conterm_hydromod_dams.txt", dtype={"STAID": str},
                   encoding="latin-1").set_index("STAID")
cat = pd.read_csv(BIG / "catalog.csv", dtype={"site_no": str})
cat = cat[cat.get("area_diff_pct").notna() & (cat.area_diff_pct < 10)]
cat = cat.set_index("site_no")
cat = cat[~cat.index.duplicated()]

CAND = []
for sid in cat.index:
    if not ((BIG / "pairs" / f"{sid}.parquet").exists()
            and (BIG / ERA5_DIR / f"{sid}.csv").exists()):
        continue
    if USE_DAYMET and not (BIG / DAYMET_DIR / f"{sid}.csv").exists():
        continue
    if USE_ZONES and not (BIG / "zone_rain" / f"{sid}.csv").exists():
        continue
    if USE_IMERG and not (BIG / "imerg" / f"{sid}.csv").exists():
        continue
    if sid not in ATT.index or sid not in dams.index:
        continue
    stor = dams.loc[sid, "STOR_NID_2009"]
    if not np.isfinite(stor) or stor > STOR_MAX:
        continue
    if ATT.loc[sid].isna().any() or not np.isfinite(cat.loc[sid, "musk_k_s"]):
        continue
    CAND.append(sid)
CAND = sorted(CAND)
if SITES_FILE:
    keep = set(open(SITES_FILE).read().split())
    CAND = [s for s in CAND if s in keep]
REF = [s for s in CAND if s in cls.index and cls.loc[s, "CLASS"] == "Ref"]
print(f"hybrid pool: {len(CAND)} basins (ref subset {len(REF)}) | {DEV}",
      flush=True)

A = ATT.loc[CAND].astype(float)
if os.environ.get("LEAN", "0") == "1":
    A = A.drop(columns=[c for c in ("SANDAVE", "BARRENNLCD06", "PLANTNLCD06",
        "ROCKDEPAVE", "SLOPE_PCT", "T_AVG_BASIN", "DEVNLCD06") if c in A.columns])
A["log_area"] = np.log(cat.loc[CAND, "drain_km2"].astype(float))              # log catchment area (km2)
A["lat"] = cat.loc[CAND, "lat"].astype(float)                                 # gauge latitude
A["lon"] = cat.loc[CAND, "lon"].astype(float)                                 # gauge longitude
A["log_k"] = np.log(cat.loc[CAND, "musk_k_s"].astype(float).clip(lower=600))  # log reach routing timescale (Muskingum K, s)
Astd = (A - A.mean()) / (A.std() + 1e-9)
N_STAT = Astd.shape[1]

TDIR = BIG / "lat_obs"
TDIR.mkdir(exist_ok=True)

# ---- DYNAMIC LSTM inputs (one column per name, in THIS exact order) ----------
DYN = [
    # precipitation (ERA5-Land total precip) + antecedent rolling means
    "tp",        # total precipitation, mm/day
    "tp3",       # 3-day mean precip   (short antecedent wetness)
    "tp7",       # 7-day mean precip
    "tp30",      # 30-day mean precip  (seasonal antecedent wetness)
    # land-surface runoff generation (HTESSEL / ERA5-Land) -- the raw signal we correct
    "sro",       # surface runoff, mm/day
    "ssro",      # sub-surface (baseflow) runoff, mm/day
    # soil-moisture state
    "sw1",       # soil water, top layer (volumetric) -- fast/near-surface store
    "sw2",       # soil water, deeper layer -- slow store
    # energy / evaporation drivers
    "t2m",       # 2 m air temperature, degC
    "ev7",       # 7-day mean actual evaporation, mm/day
    # snow
    "melt",      # snowmelt, mm/day
    "melt7",     # 7-day mean snowmelt
    "swe",       # snow water equivalent, mm (snowpack storage)
    "snowfall",  # snowfall, mm/day
    # potential ET + radiation
    "pet",       # potential evapotranspiration, mm/day
    "srad",      # surface solar radiation
    # liquid water actually reaching the soil (rain + melt) + rolling means
    "liq",       # liquid input = precip + snowmelt, mm/day
    "liq7",      # 7-day mean liquid input
    "liq30",     # 30-day mean liquid input
    "aridx",     # running aridity index = 30d PET / (30d liquid input + 1)
    # season (cyclic day-of-year encoding)
    "sin_doy",   # sin(2*pi * day_of_year / 365.25)
    "cos_doy",   # cos(2*pi * day_of_year / 365.25)
    # GEOGLOWS retrospective discharge prior (the simulation being corrected), log1p mm/day-equiv
    "gg",        # GEOGLOWS discharge, log1p
    "gg3",       # 3-day mean GEOGLOWS discharge
    "gg7",       # 7-day mean
    "gg30",      # 30-day mean
    "gg_d1",     # GEOGLOWS discharge lagged 1 day
]
if USE_DAYMET:
    # regional high-quality precip slot (Daymet, or MSWEP via DAYMET_DIR) + 3/7/30-day rolls
    DYN += ["Pd", "Pd3", "Pd7", "Pd30"]
if os.environ.get("GG_MSWEP", "0") == "1":
    # GEOGLOWS discharge re-scaled by the MSWEP/ERA5 precip ratio: injects the
    # regional precip (MSWEP) into the ROUTED discharge prior. Needs DAYMET_DIR=mswep.
    DYN += ["gg_ms", "gg_ms3", "gg_ms7", "gg_ms30"]
if USE_IMERG:
    # IMERG satellite precipitation + 3/7/30-day rolls (global, retrospective)
    DYN += ["Pi", "Pi3", "Pi7", "Pi30"]
if USE_ZONES:
    # travel-time zoned rainfall (near / middle / far thirds of the catchment) + 3/7-day rolls
    DYN += ["Pz1", "Pz1_3", "Pz1_7", "Pz2", "Pz2_3", "Pz2_7",
            "Pz3", "Pz3_3", "Pz3_7"]
if os.environ.get("GOOGLE_DIET", "0") == "1":
    # Google's information contract: raw meteorology + statics only.
    _keep = {"t2m", "pet", "srad", "sin_doy", "cos_doy",
             "Pd", "Pd3", "Pd7", "Pd30"}
    DYN = [d for d in DYN if d in _keep]
if os.environ.get("LEAN", "0") == "1":
    _drop = {"snowfall", "ev7", "srad", "pet", "t2m", "tp", "tp3", "tp30", "melt7"}
    DYN = [d for d in DYN if d not in _drop]
if NO_TP:
    DYN = [d for d in DYN if d not in ("tp", "tp3", "tp7", "tp30")]
if NO_GG:
    DYN = [d for d in DYN if not d.startswith("gg")]
if CARAVAN_CH:
    DYN = [d for d in DYN if d not in ("sro", "ssro", "ev7")]
# MINIMAL=1: statics + ERA5 runoff only (learned T-method ablation)
# MINIMAL=2: + precipitation family
_MINL = os.environ.get("MINIMAL", "0")
if _MINL == "1":
    DYN = [d for d in DYN if d in ("sro", "ssro", "sin_doy", "cos_doy")]
elif _MINL == "2":
    DYN = [d for d in DYN if d in ("sro", "ssro", "tp", "tp3", "tp7", "tp30",
                                   "sin_doy", "cos_doy")]

DATA = {}
for i, sid in enumerate(CAND):
    try:
        f = pd.read_csv(BIG / ERA5_DIR / f"{sid}.csv", index_col=0,
                        parse_dates=True)
        if os.environ.get("TMIN"):
            f = f[f.index >= pd.Timestamp(os.environ["TMIN"])]
        pr = pd.read_parquet(BIG / "pairs" / f"{sid}.parquet")
        pr.index = pd.to_datetime(pr.index)
        area = float(cat.loc[sid, "drain_km2"])
        mm = area * 1e6 / 1000 / 86400
        k = float(cat.loc[sid, "musk_k_s"]) / 86400
        q_o = pr.q_obs_cms.reindex(f.index)
        if q_o.notna().sum() < 3650:
            continue
        tp = TDIR / f"{sid}.parquet"
        if tp.exists():
            lat_o = pd.read_parquet(tp)["lat_mm"]
        else:
            qv = (q_o / mm).interpolate(limit=7).dropna()
            n = len(qv)
            Ar = conv_matrix(kernel(k, n), n)
            lat_o = pd.Series(invert(qv.to_numpy(), Ar, lam=0.05, iters=600),
                              qv.index, name="lat_mm")
            lat_o.to_frame().to_parquet(tp)
        gg = (pr.q_gg_cms.reindex(f.index) / mm).clip(lower=0)
        if CORR_RUNOFF and sid in _TP.index:
            _p = _TP.loc[sid]
            _sv = f.sro_mm.fillna(0).to_numpy(); _bv = f.ssro_mm.fillna(0).to_numpy()
            _sro_in = pd.Series(np.clip(_retime(_sv, _acf_T(_sv), _p.Tf_o), 0, None)*_p.g_f, f.index)
            _ssro_in = pd.Series(np.clip(_retime(_bv, _acf_T(_bv), _p.Ts_o), 0, None)*_p.g_s, f.index)
        else:
            _sro_in, _ssro_in = f.sro_mm, f.ssro_mm
        if CARAVAN_CH:
            melt = (-f.swe_mm.diff()).clip(lower=0).fillna(0)
            f = f.assign(snowfall_mm=f.tp_mm.where(f.t2m_c < 0.5, 0.0))
        else:
            melt = f.melt_mm.clip(lower=0).fillna(0)
        if USE_DAYMET:
            Pd = pd.read_csv(BIG / DAYMET_DIR / f"{sid}.csv", index_col=0,
                             parse_dates=True)["prcp_mm"].reindex(f.index).fillna(0)
        else:
            Pd = None
        if USE_IMERG:
            Pi = pd.read_csv(BIG / "imerg" / f"{sid}.csv", index_col=0,
                             parse_dates=True)["prcp_mm"].reindex(f.index).fillna(0)
            _imerg_ch = {"Pi": Pi, "Pi3": Pi.rolling(3, min_periods=1).mean(),
                         "Pi7": Pi.rolling(7, min_periods=1).mean(),
                         "Pi30": Pi.rolling(30, min_periods=1).mean()}
        if USE_ZONES:
            _zr = pd.read_csv(BIG / "zone_rain" / f"{sid}.csv", index_col=0,
                              parse_dates=True)
            _zone_ch = {}
            for _z in (1, 2, 3):
                _s = _zr[f"z{_z}"].reindex(f.index).fillna(0)
                _zone_ch[f"Pz{_z}"] = _s
                _zone_ch[f"Pz{_z}_3"] = _s.rolling(3, min_periods=1).mean()
                _zone_ch[f"Pz{_z}_7"] = _s.rolling(7, min_periods=1).mean()
        base_p = Pd if Pd is not None else \
            (melt * 0 if NO_TP else f.tp_mm.fillna(0))
        liq = base_p + melt
        doy = f.index.dayofyear.to_numpy()
        # gg_mswep: scale GEOGLOWS discharge by the 7-day MSWEP/ERA5 precip ratio
        if os.environ.get("GG_MSWEP", "0") == "1" and Pd is not None:
            _msr = ((Pd.rolling(7, min_periods=1).mean() + 0.5)
                    / (f.tp_mm.rolling(7, min_periods=1).mean() + 0.5)).clip(0.3, 3.0)
            gg_ms = (gg * _msr).clip(lower=0)
        else:
            gg_ms = gg
        X = pd.DataFrame({
            "tp": f.tp_mm, "tp3": f.tp_mm.rolling(3, min_periods=1).mean(),
            "tp7": f.tp_mm.rolling(7, min_periods=1).mean(),
            "tp30": f.tp_mm.rolling(30, min_periods=1).mean(),
            "sro": _sro_in, "ssro": _ssro_in, "sw1": f.sw1, "sw2": f.sw2,
            "t2m": f.t2m_c, "ev7": f.ev_mm.rolling(7, min_periods=1).mean(),
            "melt": melt, "melt7": melt.rolling(7, min_periods=1).mean(),
            "swe": f.swe_mm, "snowfall": f.snowfall_mm, "pet": f.pet_mm,
            "srad": f.srad, "liq": liq,
            "liq7": liq.rolling(7, min_periods=1).mean(),
            "liq30": liq.rolling(30, min_periods=1).mean(),
            "aridx": f.pet_mm.rolling(30, min_periods=1).mean()
            / (liq.rolling(30, min_periods=1).mean() + 1.0),
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "gg": np.log1p(gg),
            "gg3": np.log1p(gg.rolling(3, min_periods=1).mean()),
            "gg7": np.log1p(gg.rolling(7, min_periods=1).mean()),
            "gg30": np.log1p(gg.rolling(30, min_periods=1).mean()),
            "gg_d1": np.log1p(gg.shift(1)),
            **({"Pd": Pd, "Pd3": Pd.rolling(3, min_periods=1).mean(),
                "Pd7": Pd.rolling(7, min_periods=1).mean(),
                "Pd30": Pd.rolling(30, min_periods=1).mean()}
               if USE_DAYMET else {}),
            **({"gg_ms": np.log1p(gg_ms),
                "gg_ms3": np.log1p(gg_ms.rolling(3, min_periods=1).mean()),
                "gg_ms7": np.log1p(gg_ms.rolling(7, min_periods=1).mean()),
                "gg_ms30": np.log1p(gg_ms.rolling(30, min_periods=1).mean())}
               if os.environ.get("GG_MSWEP", "0") == "1" else {}),
            **(_zone_ch if USE_ZONES else {}),
            **(_imerg_ch if USE_IMERG else {})},
            index=f.index).fillna(0.0)
        if GSTYLE:
            y = np.log1p((q_o / mm).clip(lower=0))
        else:
            y = np.log1p(lat_o.clip(lower=0)).reindex(f.index)
        DATA[sid] = dict(
            idx=f.index, X=X[DYN].to_numpy(np.float32),
            y=y.fillna(0).to_numpy(np.float32),
            valid=(y.notna() & q_o.notna()).to_numpy(),
            qcms=q_o.to_numpy(np.float32), mm=mm, k=k,
            ggcms=pr.q_gg_cms.reindex(f.index).to_numpy(np.float32),
            ystd=np.float32(max(np.nanstd(np.expm1(y.to_numpy())), 0.05)))
        if i % 300 == 0:
            print(f"tensors {i}/{len(CAND)}", flush=True)
    except Exception as e:
        print(f"skip {sid}: {str(e)[:60]}", flush=True)
CAND = [s for s in CAND if s in DATA]
REF = [s for s in REF if s in DATA]
print(f"tensors built: {len(CAND)}", flush=True)
N_DYN = len(DYN)

snowp = A.loc[CAND, "SNOW_PCT_PRECIP"]
aridp = A.loc[CAND, "PET"] / A.loc[CAND, "PPTAVG_BASIN"].clip(lower=1)
strat = (snowp >= 20).astype(int) * 10 + pd.qcut(aridp, 3, labels=False,
                                                 duplicates="drop")
folds = [[] for _ in range(6)]
for _, grp in pd.Series(CAND, index=CAND).groupby(strat):
    for i, g in enumerate(RNG.permutation(grp.index)):
        folds[i % 6].append(g)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        if LIQUID:
            from ncps.torch import CfC
            self.lstm = CfC(N_DYN + N_STAT, HID, batch_first=True)
        else:
            self.lstm = nn.LSTM(N_DYN + N_STAT, HID, batch_first=True)
        if ENH_EMA:
            self.head2 = nn.Linear(HID, 2)        # fast, slow (pre-store)
            self.amlp = nn.Sequential(nn.Linear(N_STAT, 16), nn.Tanh(),
                                      nn.Linear(16, 1))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(HID, 3 if (GSTYLE or ARCH_ONLY) else 1)
        if DPL_LEARNK:                 # learn a per-basin K multiplier from statics
            self.kmlp = nn.Sequential(nn.Linear(N_STAT, 16), nn.Tanh(), nn.Linear(16, 1))

    def kmult(self, x):
        if not DPL_LEARNK:
            return None
        return torch.exp(torch.tanh(self.kmlp(x[:, 0, -N_STAT:])[:, 0]) * 0.7)  # ~[0.5, 2]

    def forward(self, x):
        o, _ = self.lstm(x)
        if ENH_EMA:
            h2 = self.head2(self.drop(o))
            fast = nn.functional.softplus(h2[..., 0])
            slow_in = nn.functional.softplus(h2[..., 1])
            # store time-constant from statics (same at every timestep)
            a = torch.sigmoid(self.amlp(x[:, 0, -N_STAT:]))[:, 0] * 0.145 + 0.85
            seq = []
            state = slow_in[:, 0]
            for t in range(slow_in.shape[1]):
                state = a * state + (1 - a) * slow_in[:, t]
                seq.append(state)
            slow = torch.stack(seq, 1)
            return torch.log1p(fast + slow)
        h = self.head(self.drop(o))
        return h if (GSTYLE or ARCH_ONLY) else h[..., 0]


def droute(lat_seq, T, tail=120):
    """Differentiable linear reservoir (matches mb_core.reservoir): route the
    lateral sequence (B,L) through a per-basin store T (days) -> discharge (B,).
    Returns the routed discharge at the LAST timestep. Only the last `tail` days
    are routed (reservoir memory << SEQ) for speed; storage warm-started from the
    pre-tail lateral mean."""
    T = T.clamp(min=0.05)
    a = torch.exp(-1.0 / T)                                 # (B,)
    L = lat_seq.shape[1]; s = max(0, L - tail)
    S = T * lat_seq[:, :max(s, 1)].mean(1)                  # warm-start storage
    q = lat_seq[:, s]                                       # init
    for t in range(s, L):
        x = lat_seq[:, t]
        S_next = a * S + T * (1 - a) * x
        q = x + (S - S_next)
        S = S_next
    return q.clamp(min=0)                                   # discharge at last step (B,)


def run_fold(fi, test):
    train = [g for g in CAND if g not in test]
    Xtr = np.concatenate([DATA[g]["X"][::7] for g in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    stat = {g: Astd.loc[g].to_numpy(np.float32) for g in CAND}
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    def batch(nb=256):
        xs, ys, ss, y30, qd, kk, yseq = [], [], [], [], [], [], []
        for g in TRNG.choice(train, size=nb):
            d = DATA[g]
            ok = np.where(d["valid"] & (np.arange(len(d["y"])) >= SEQ))[0]
            if not len(ok):
                continue
            t = int(TRNG.choice(ok))
            xw = (d["X"][t - SEQ + 1:t + 1] - mu) / sd
            sw = np.repeat(stat[g][None, :], SEQ, 0)
            xs.append(np.concatenate([xw, sw], 1))
            ys.append(d["y"][t]); ss.append(d["ystd"])
            y30.append(np.log1p(np.expm1(d["y"][t-29:t+1]).mean())
                       if ENH_MS else 0.0)
            # DPL: observed discharge at t (normalized to mm/day-equiv, log) + basin K
            qd.append(np.log1p(max(float(d["qcms"][t]), 0.0) / d["mm"]) if DPL else 0.0)
            kk.append(float(d["k"]) if DPL else 1.0)
            # VAR_LOSS: target lateral over the window (for dispersion matching)
            yseq.append(d["y"][t - SEQ + 1:t + 1] if (VAR_LOSS or TIME_LOSS or VOL_ANCHOR) else np.zeros(1, np.float32))
        T_ = lambda a: torch.tensor(np.asarray(a), device=DEV)
        return T_(np.stack(xs)), T_(np.array(ys, np.float32)), \
            T_(np.array(ss, np.float32)), T_(np.array(y30, np.float32)), \
            T_(np.array(qd, np.float32)), T_(np.array(kk, np.float32)), \
            (T_(np.stack(yseq)) if (VAR_LOSS or TIME_LOSS or VOL_ANCHOR) else None)

    net.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for _ in range(STEPS):
            xb, yb, sb, y30b, qdb, kkb, yseqb = batch()
            if GSTYLE or ARCH_ONLY:
                h = net(xb)[:, -1, :]
                mu_p, log_b = h[:, 0], h[:, 1].clamp(-6, 3)
                tau = torch.sigmoid(h[:, 2]) * 0.8 + 0.1
                e = yb - mu_p
                loss = (torch.where(e >= 0, tau * e, (tau - 1) * e)
                        / torch.exp(log_b) + log_b).mean()
            else:
                seq_out = net(xb)
                out = seq_out[:, -1]
                lat_lin = torch.expm1(out.clamp(max=30))
                # PEAK_W>0: extreme-weighted loss — up-weight high-flow samples
                # (targets rare-event detection, Nearing-style objective)
                _pw = float(os.environ.get("PEAK_W", "0"))
                if _pw > 0:
                    w = 1.0 + _pw * (torch.expm1(yb) / (sb + 0.1)).clamp(0, 5)
                    loss = (w * (out - yb) ** 2).mean() \
                        + 0.3 * (w * ((lat_lin - torch.expm1(yb)) / (sb + 0.1)) ** 2).mean()
                else:
                    loss = ((out - yb) ** 2).mean() \
                        + 0.3 * (((lat_lin - torch.expm1(yb)) / (sb + 0.1)) ** 2).mean()
                if ENH_MS:
                    m30 = torch.expm1(seq_out[:, -30:].clamp(max=30)).mean(1)
                    t30 = torch.expm1(y30b.clamp(max=30))
                    loss = loss + 0.3 * (((m30 - t30) / (sb + 0.1)) ** 2).mean()
                if DPL:
                    # route predicted lateral seq -> discharge at t, loss vs observed
                    lat_seq = torch.expm1(seq_out.clamp(max=30))
                    T_eff = kkb * (net.kmult(xb) if DPL_LEARNK else 1.0)
                    q_pred = droute(lat_seq, T_eff)
                    loss = loss + DPL_W * ((torch.log1p(q_pred) - qdb) ** 2).mean()
                if VAR_LOSS or TIME_LOSS or VOL_ANCHOR:
                    lp = torch.expm1(seq_out.clamp(max=30))
                    lt = torch.expm1(yseqb.clamp(max=30))
                if VAR_LOSS:
                    # penalize under-dispersion: predicted lateral-seq std vs target std
                    sp = lp.std(dim=1); st = lt.std(dim=1)
                    if VAR_CV:
                        # mean-preserving: match CV (sigma/mu) so raising variance does
                        # not inflate the mean (fixes the arid beta overshoot)
                        cvp = sp / (lp.mean(dim=1) + 0.1); cvt = st / (lt.mean(dim=1) + 0.1)
                        loss = loss + VAR_W * ((cvp - cvt) ** 2).mean()
                    else:
                        loss = loss + VAR_W * (((sp - st) / (sb + 0.1)) ** 2).mean()
                if TIME_LOSS:
                    # windowed Pearson correlation -> directly attacks the KGE r term
                    lpc = lp - lp.mean(dim=1, keepdim=True)
                    ltc = lt - lt.mean(dim=1, keepdim=True)
                    num = (lpc * ltc).sum(dim=1)
                    den = torch.sqrt((lpc ** 2).sum(dim=1) * (ltc ** 2).sum(dim=1) + 1e-6)
                    rho = num / (den + 1e-6)
                    loss = loss + TIME_W * (1.0 - rho).mean()
                if VOL_ANCHOR:
                    # pin window-mean volume -> penalize (beta-1)^2, blocks the timing/CV
                    # loss from inflating the mean (cures the arid beta overshoot)
                    mp = lp.mean(dim=1); mt = lt.mean(dim=1)
                    loss = loss + VOL_W * (((mp - mt) / (mt + 0.1)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tot += float(loss)
        sched.step()
        if ep % 4 == 0:
            print(f"  fold {fi} ep {ep} loss {tot/STEPS:.4f}", flush=True)

    if SAVE_MODELS:
        _md = RESULTS / os.environ.get("MODELS_DIR", "champ_models")
        _md.mkdir(exist_ok=True)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                    "test": list(test), "dyn": DYN,
                    "stat_cols": list(Astd.columns),
                    "amu": A.mean().to_numpy(np.float32),
                    "asd": (A.std() + 1e-9).to_numpy(np.float32)},
                   _md / f"fold{fi}.pt")
    net.eval()
    rows = []
    with torch.no_grad():
        for g in test:
            d = DATA[g]
            xw = (d["X"] - mu) / sd
            sw = np.repeat(stat[g][None, :], len(xw), 0)
            xt = torch.tensor(np.concatenate([xw, sw], 1)[None], device=DEV)
            o_ = net(xt)[0].cpu().numpy()
            if ARCH_ONLY:
                lat = np.expm1(np.clip(o_[:, 0], None, 30)).clip(0)
                sim = pd.Series(reservoir(lat * d["mm"], d["k"]), d["idx"])
            elif GSTYLE:
                sim = pd.Series(np.expm1(np.clip(o_[:, 0], None, 30)).clip(0)
                                * d["mm"], d["idx"])
            else:
                lat = np.expm1(np.clip(o_, None, 30)).clip(0)
                sim = pd.Series(reservoir(lat * d["mm"], d["k"]), d["idx"])
            q_obs = pd.Series(d["qcms"], d["idx"])
            gg_cms = pd.Series(d["ggcms"], d["idx"])
            yrs = sorted(set(q_obs.dropna().index.year))
            if len(yrs) < 6:
                continue
            VAL = slice(str(yrs[len(yrs) // 2]), str(yrs[-1]))
            if os.environ.get("SAVE_LAT", "1") == "1" and not (GSTYLE):
                _tag = (("liquid" if LIQUID else "arch" if ARCH_ONLY else "ours")
                        + ("_dm" if USE_DAYMET else "_era5")
                        + ("_ms" if ENH_MS else "") + ("_ema" if ENH_EMA else "")
                        + (f"_{SYN}" if SYN else "") + ("_notp" if NO_TP else "")
                        + ("_zones" if USE_ZONES else ""))
                _d = RESULTS / f"latpred_{_tag}"
                _d.mkdir(exist_ok=True)
                pd.Series(lat, d["idx"], name="lat_pred_mm").round(4)\
                    .to_frame().to_csv(_d / f"{g}.csv")
            a = panel(sim.loc[VAL], q_obs.loc[VAL])
            b = panel(gg_cms.loc[VAL], q_obs.loc[VAL])
            tv = timing_volume(sim.loc[VAL], q_obs.loc[VAL])
            tvg = {f"{k2}_gg": v for k2, v in
                   timing_volume(gg_cms.loc[VAL], q_obs.loc[VAL]).items()}
            rows.append(dict(gid=g, fold=fi, KGE=a["KGE"], r=a["r"],
                             logNSE=a["logNSE"], KGE_gg=b["KGE"],
                             logNSE_gg=b["logNSE"], **tv, **tvg))
    return rows


all_rows = []
for fi, test in enumerate(folds):
    all_rows += run_fold(fi, test)
    T = pd.DataFrame(all_rows)
    print(f"fold {fi+1}/6 | hybrid {T.KGE.median():+.3f} "
          f"vs GEOGLOWS {T.KGE_gg.median():+.3f}", flush=True)

T = pd.DataFrame(all_rows)
_parts = ["big",
          "liquid" if LIQUID else "gstyle" if GSTYLE else
          "arch" if ARCH_ONLY else "ours",
          "msema" if (ENH_MS and ENH_EMA) else "ms" if ENH_MS else
          "ema" if ENH_EMA else "",
          SYN if SYN else "dm" if USE_DAYMET else "era5",
          "notp" if NO_TP else "",
          "corr" if CORR_RUNOFF else "",
          "caravanch" if CARAVAN_CH else "nogg" if NO_GG else "",
          "zones" if USE_ZONES else "",
          "imerg" if USE_IMERG else "",
          (f"minimal{_MINL}" if _MINL != "0" else ""),
          ("pilot" if "daymet_pilot" in SITES_FILE
           else Path(SITES_FILE).stem) if SITES_FILE else "full"]
_name = "_".join(p for p in _parts if p) + f"_{SEED}.csv"
T.round(4).to_csv(RESULTS / _name, index=False)
print("results ->", _name, flush=True)
print(f"\n=== " + ("LIQUID CfC" if LIQUID else "GOOGLE-ARCH + OUR I/O" if ARCH_ONLY else "GOOGLE-STYLE ARCH" if GSTYLE else "HYBRID AT SCALE") + " (n={len(T)} held-out basins) ===")
for label, S in (("all screened", T), ("reference-class", T[T.gid.isin(REF)])):
    print(f"-- {label} (n={len(S)})")
    print(f"   hybrid  : KGE {S.KGE.median():+.3f} "
          f"[{S.KGE.quantile(.25):+.2f},{S.KGE.quantile(.75):+.2f}] "
          f"r {S.r.median():+.3f} logNSE {S.logNSE.median():+.3f} "
          f">0.7 {(S.KGE > .7).mean():.0%}")
    print(f"   GEOGLOWS: KGE {S.KGE_gg.median():+.3f} "
          f"logNSE {S.logNSE_gg.median():+.3f}")
    if "beta" in S:
        print(f"   volume: beta {S.beta.median():.2f} (gg {S.beta_gg.median():.2f}) "
              f"| variability gamma {S.gamma.median():.2f} (gg {S.gamma_gg.median():.2f})")
        print(f"   timing: xcorr lag {S.xlag_d.median():+.1f}d (gg {S.xlag_d_gg.median():+.1f}d) "
              f"| median |peak lag| {S.abs_peak_lag_d.median():.1f}d "
              f"(gg {S.abs_peak_lag_d_gg.median():.1f}d)")
print("refs: 344-basin lateral 0.699 | product-only@3235 0.363")
print("BIG_LIQUID_DONE" if LIQUID else "BIG_ARCHONLY_DONE" if ARCH_ONLY else "BIG_GSTYLE_DONE" if GSTYLE else "BIG_HYBRID_DONE", flush=True)
