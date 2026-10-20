"""Evaluation metrics for the SSRO store experiment.

KGE follows Kling et al. (2012): gamma is the ratio of coefficients of
variation, so bias (beta) and variability (gamma) are independent.
"""
import numpy as np
import pandas as pd


def _paired(sim: pd.Series, obs: pd.Series):
    df = pd.concat([sim, obs], axis=1, keys=["sim", "obs"]).dropna()
    return df["sim"].to_numpy(), df["obs"].to_numpy()


def kge2012(sim: pd.Series, obs: pd.Series) -> dict:
    """KGE-2012 and its components r (correlation), beta (mean ratio),
    gamma (CV ratio)."""
    s, o = _paired(sim, obs)
    if len(s) < 2:
        return {"kge": np.nan, "r": np.nan, "beta": np.nan, "gamma": np.nan}
    r = np.corrcoef(s, o)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std(ddof=1) / s.mean()) / (o.std(ddof=1) / o.mean())
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return {"kge": float(kge), "r": float(r), "beta": float(beta),
            "gamma": float(gamma)}


def nse(sim: pd.Series, obs: pd.Series) -> float:
    s, o = _paired(sim, obs)
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def log_nse(sim: pd.Series, obs: pd.Series, eps_frac: float = 0.01) -> float:
    """NSE on log flows; eps = eps_frac * mean(obs) guards against zeros
    (standard practice, e.g. Pushpalatha et al. 2012)."""
    s, o = _paired(sim, obs)
    eps = eps_frac * o.mean()
    ls, lo = np.log(s + eps), np.log(o + eps)
    return float(1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2))


def seasonal_kge(sim: pd.Series, obs: pd.Series) -> dict:
    """KGE over the 4 highest- and 4 lowest-flow calendar months, ranked by
    the station's observed monthly climatology (per Chancay et al.)."""
    clim = obs.groupby(obs.index.month).mean().sort_values()
    low4, high4 = list(clim.index[:4]), list(clim.index[-4:])
    out = {}
    for tag, months in (("high4", high4), ("low4", low4)):
        m = obs.index.month.isin(months)
        out[tag] = kge2012(sim[m], obs[m])
        out[tag]["months"] = months
    return out


def lyne_hollick_bfi(q: pd.Series, alpha: float = 0.925,
                     passes: int = 3) -> float:
    """Baseflow index from the Lyne-Hollick recursive digital filter
    (3 passes forward/backward/forward, standard alpha=0.925).
    NaNs are interpolated first; result is BFI = sum(qb)/sum(q)."""
    qv = q.interpolate(limit_direction="both").to_numpy(dtype=float)
    qb = qv.copy()
    for p in range(passes):
        x = qb if p == 0 else qb  # filter operates on previous pass output
        x = x[::-1] if p % 2 else x
        qf = np.zeros_like(x)  # quickflow
        for i in range(1, len(x)):
            qf[i] = alpha * qf[i - 1] + 0.5 * (1 + alpha) * (x[i] - x[i - 1])
            qf[i] = max(qf[i], 0.0)
        base = np.clip(x - qf, 0.0, x)
        qb = base[::-1] if p % 2 else base
    return float(np.sum(qb) / np.sum(qv))


def fdc_quantile_map(sim_cal: pd.Series, obs_cal: pd.Series,
                     sim_apply: pd.Series, n_q: int = 200) -> pd.Series:
    """Monthly flow-duration-curve quantile mapping (Sanchez Lozano style):
    for each calendar month, map simulated values through the sim-FDC ->
    obs-FDC transform fitted on the calibration split."""
    out = pd.Series(index=sim_apply.index, dtype=float)
    probs = np.linspace(0.001, 0.999, n_q)
    for month in range(1, 13):
        sc = sim_cal[sim_cal.index.month == month].dropna()
        oc = obs_cal[obs_cal.index.month == month].dropna()
        sa = sim_apply[sim_apply.index.month == month]
        if len(sc) < 30 or len(oc) < 30:
            out[sa.index] = sa
            continue
        sim_q = np.quantile(sc, probs)
        obs_q = np.quantile(oc, probs)
        # interpolate each simulated value's nonexceedance prob on the sim
        # FDC, then read the observed FDC at that prob (linear, clamped).
        mapped = np.interp(sa.to_numpy(), sim_q, obs_q,
                           left=obs_q[0], right=obs_q[-1])
        out[sa.index] = mapped
    return out
