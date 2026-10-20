"""Multi-basin core: the frozen component-correction pipeline, per basin.

Self-contained (no cross-module globals). For one basin, given:
    obs Q [m³/s], SRO/SSRO/tp [mm/day basin means], area [km²],
    routing k [days], chronological cal/val split
it (1) derives LH/ECK component targets, (2) shape-selects Tf and (Ts, phi)
on calibration only, (3) volume-matches component gains af/as on calibration,
(4) routes, (5) evaluates raw / QM / shape / shape+gains on validation.
Returns a flat dict of parameters + metrics for the regionalization stage.
"""
import numpy as np
import pandas as pd

from metrics import fdc_quantile_map, kge2012, log_nse, lyne_hollick_bfi, nse

TF_GRID = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
TS_GRID = [0, 2, 5, 10, 15, 20, 30, 45, 60, 90]
PHI_GRID = [0.25, 0.5, 0.75, 1.0]


# ------------------------------------------------------------ primitives --
def reservoir(x: np.ndarray, T: float, s0: float = None):
    if T <= 0:
        return x.astype(float).copy()
    a = np.exp(-1.0 / T)
    if s0 is None:
        s0 = T * np.nanmean(x[:365])
    q = np.empty_like(x, dtype=float)
    S = s0
    for i, xi in enumerate(x):
        S_next = a * S + T * (1 - a) * xi
        q[i] = xi + (S - S_next)
        S = S_next
    return q


def lyne_hollick(q: pd.Series, alpha=0.925, passes=3) -> pd.Series:
    v = q.interpolate(limit_direction="both").to_numpy(float)
    qb = v.copy()
    for p in range(passes):
        x = qb[::-1] if p % 2 else qb
        qf = np.zeros_like(x)
        for i in range(1, len(x)):
            qf[i] = alpha * qf[i - 1] + 0.5 * (1 + alpha) * (x[i] - x[i - 1])
            qf[i] = max(qf[i], 0.0)
        base = np.clip(x - qf, 0.0, x)
        qb = base[::-1] if p % 2 else base
    return pd.Series(qb, index=q.index)


def eckhardt(q: pd.Series, a: float, bfi_max=0.80) -> pd.Series:
    v = q.interpolate(limit_direction="both").to_numpy(float)
    b = np.empty_like(v)
    b[0] = v[0] * bfi_max
    for i in range(1, len(v)):
        b[i] = min(((1 - bfi_max) * a * b[i - 1]
                    + (1 - a) * bfi_max * v[i]) / (1 - a * bfi_max), v[i])
    return pd.Series(np.clip(b, 0, None), index=q.index)


def acf_stats(x: pd.Series, maxlag=120) -> dict:
    v = x.dropna().to_numpy(float)
    v = v - v.mean()
    den = np.sum(v * v)
    rho = np.array([np.sum(v[:-k] * v[k:]) / den
                    for k in range(1, maxlag + 1)])
    neg = np.where(rho < 0)[0]
    stop = int(neg[0]) if len(neg) else maxlag
    return {"lag1": float(rho[0]), "integral": float(0.5 + rho[:stop].sum())}


# ------------------------------------------------------ shape machinery ---
def fast_events(qfast: pd.Series, span) -> list:
    qf = qfast.loc[span].dropna()
    if len(qf) < 400:
        return []
    thr = qf[qf > 0].quantile(0.80)
    v, peaks = qf.to_numpy(), []
    for i in range(3, len(v) - 6):
        if v[i] == v[i - 3:i + 4].max() and v[i] > thr:
            if not peaks or (qf.index[i] - peaks[-1]).days >= 7:
                peaks.append(qf.index[i])
    return [qf.index[(qf.index >= p - pd.Timedelta(days=3))
                     & (qf.index <= p + pd.Timedelta(days=6))] for p in peaks]


def fast_score(x: pd.Series, target: pd.Series, wins) -> float:
    plags, nrmses, xlags = [], [], []
    for w in wins:
        xs = x.reindex(w).to_numpy(float)
        ts = target.reindex(w).to_numpy(float)
        if np.isnan(xs).any() or np.isnan(ts).any():
            continue
        if xs.max() <= 1e-6 or ts.max() <= 1e-6:
            continue
        xn, tn = xs / xs.max(), ts / ts.max()
        plags.append(int(np.argmax(xn)) - int(np.argmax(tn)))
        nrmses.append(np.sqrt(np.mean((xn - tn) ** 2)))
        xa, ta = xn - xn.mean(), tn - tn.mean()
        cc = []
        for L in range(-4, 5):
            a_, b_ = (xa[L:], ta[:len(ta) - L]) if L >= 0 else (xa[:L], ta[-L:])
            cc.append(np.corrcoef(a_, b_)[0, 1] if len(a_) > 3 else np.nan)
        xlags.append(range(-4, 5)[int(np.nanargmax(cc))])
    if len(nrmses) < 5:
        return np.nan
    return (abs(np.median(plags)) + abs(np.median(xlags))
            + 2 * float(np.median(nrmses)))


def slow_windows(qslow: pd.Series, tp: pd.Series, span, min_len=10) -> list:
    qs = qslow.loc[span].dropna()
    dry = (tp.reindex(qs.index).fillna(0) <= 0.5)
    dec = (qs.diff() < 0.02 * qs) & dry & dry.shift(1).fillna(False)
    wins, start, flags = [], None, dec.to_numpy()
    for i, f in enumerate(flags):
        if f and start is None:
            start = i - 1
        elif not f and start is not None:
            if i - start >= min_len and qs.iloc[start] > qs.iloc[i - 1]:
                wins.append(qs.index[start:i])
            start = None
    return wins


def slow_score(x: pd.Series, target: pd.Series, wins) -> float:
    dTs, nrmses, d10, d20 = [], [], [], []
    for w in wins:
        xs = x.reindex(w).to_numpy(float)
        ts = target.reindex(w).to_numpy(float)
        if np.isnan(xs).any() or np.isnan(ts).any():
            continue
        xs, ts = xs.clip(1e-6), ts.clip(1e-6)
        xn, tn = xs / xs[0], ts / ts[0]
        t = np.arange(len(w), dtype=float)
        slx = np.polyfit(t, np.log(xn), 1)[0]
        slt = np.polyfit(t, np.log(tn), 1)[0]
        if slx < 0 and slt < 0:
            dTs.append(-1 / slx + 1 / slt)
        nrmses.append(np.sqrt(np.mean((xn - tn) ** 2)))
        if len(w) > 10:
            d10.append(xn[10] - tn[10])
        if len(w) > 20:
            d20.append(xn[20] - tn[20])
    if len(nrmses) < 4:
        return np.nan
    s = (abs(np.median(dTs)) / 10 if dTs else 5.0) \
        + 2 * float(np.median(nrmses))
    s += abs(np.median(d10)) if d10 else 0.5
    s += abs(np.median(d20)) if d20 else 0.5
    return float(s)


# ------------------------------------------------------------- evaluation -
def panel(sim: pd.Series, o: pd.Series) -> dict:
    both = pd.concat([sim, o], axis=1, keys=["s", "o"]).dropna()
    if len(both) < 365:
        return {}
    s, ov = both["s"], both["o"]
    k = kge2012(s, ov)
    out = {"KGE": k["kge"], "r": k["r"], "beta": k["beta"],
           "gamma": k["gamma"], "NSE": nse(s, ov), "logNSE": log_nse(s, ov),
           "PBIAS": float(100 * (s - ov).sum() / ov.sum()),
           "BFI": lyne_hollick_bfi(s),
           "p5p50": float(np.percentile(s, 5) / np.percentile(s, 50)),
           **{f"acf_{k2}": v for k2, v in acf_stats(s).items()}}
    for nm, a, b in (("q0-10", 0, .10), ("q90-100", .90, 1.0)):
        m = (ov > ov.quantile(a)) & (ov <= ov.quantile(b))
        out[f"bias_{nm}"] = float(s[m].mean() / ov[m].mean())
    return out


# ------------------------------------------------------------- pipeline ---
def run_basin(gid: str, q_obs: pd.Series, sro: pd.Series, ssro: pd.Series,
              tp: pd.Series, area_km2: float, k_route_days: float) -> dict:
    """Full frozen pipeline for one basin. Series share a daily index."""
    idx = sro.index
    q_obs = q_obs.reindex(idx)
    mm2cms = area_km2 * 1e6 / 1000 / 86400
    to_mm = 86400 / (area_km2 * 1e6) * 1000

    good = q_obs.dropna()
    years = sorted(set(good.index.year))
    if len(years) < 8:
        return {"gauge_id": gid, "error": "record too short"}
    mid = years[len(years) // 2]
    CAL = slice(str(years[0]), f"{mid - 1}-12-31")
    VAL = slice(str(mid), str(years[-1]))

    r_obs = q_obs * to_mm
    # recession constant for Eckhardt a
    a_eck = np.exp(-1 / 25.0)
    qslow_lh = lyne_hollick(q_obs.dropna()) * to_mm
    qslow_ec = eckhardt(q_obs.dropna(), a_eck) * to_mm
    qfast_lh = (r_obs - qslow_lh).clip(lower=0)

    out = {"gauge_id": gid, "cal_end": CAL.stop, "n_years": len(years),
           "bfi_lh": float(qslow_lh.loc[CAL].sum() / r_obs.loc[CAL].sum())}

    # --- Tf (LH target, shape only, calibration) ---
    wins_f = fast_events(qfast_lh, CAL)
    scores = {Tf: fast_score(pd.Series(reservoir(sro.to_numpy(), Tf), idx),
                             qfast_lh, wins_f) for Tf in TF_GRID}
    scores = {k2: v for k2, v in scores.items() if np.isfinite(v)}
    if not scores:
        return {**out, "error": "no fast events"}
    out["Tf"] = float(min(scores, key=scores.get))
    out["n_fast_events"] = len(wins_f)

    # --- (Ts, phi) (LH target) + ECK sensitivity ---
    for tag, qs_t in (("", qslow_lh), ("_eck", qslow_ec)):
        wins_s = slow_windows(qs_t, tp, CAL)
        sscore = {}
        for Ts in TS_GRID:
            store = reservoir(ssro.to_numpy(), Ts)
            for phi in (PHI_GRID if Ts > 0 else [1.0]):
                s = pd.Series((1 - phi) * ssro.to_numpy() + phi * store, idx)
                sscore[(Ts, phi)] = slow_score(s, qs_t, wins_s)
        sscore = {k2: v for k2, v in sscore.items() if np.isfinite(v)}
        if not sscore:
            if tag == "":
                return {**out, "error": "no slow windows"}
            continue
        (ts_b, phi_b) = min(sscore, key=sscore.get)
        out[f"Ts{tag}"], out[f"phi{tag}"] = float(ts_b), float(phi_b)
        if tag == "":
            out["n_slow_windows"] = len(wins_s)

    # --- build shape components, route, gains -----------------------------
    def route(x_mm):
        lat = x_mm * mm2cms
        return pd.Series(reservoir(lat, k_route_days), idx)

    fast_shape = reservoir(sro.to_numpy(), out["Tf"])
    store = reservoir(ssro.to_numpy(), out["Ts"])
    slow_shape = (1 - out["phi"]) * ssro.to_numpy() + out["phi"] * store
    qf_r, qs_r = route(fast_shape), route(slow_shape)

    oc = q_obs.loc[CAL].dropna()
    qfast_cms = (qfast_lh / to_mm).reindex(oc.index)
    qslow_cms = (qslow_lh / to_mm).reindex(oc.index)
    out["af"] = float(qfast_cms.sum() / qf_r.loc[CAL].reindex(oc.index).sum())
    out["as"] = float(qslow_cms.sum() / qs_r.loc[CAL].reindex(oc.index).sum())

    # --- configurations, validation ---------------------------------------
    raw = route(sro.to_numpy() + ssro.to_numpy())
    cfgs = {
        "raw": raw,
        "qm": fdc_quantile_map(raw.loc[CAL], q_obs.loc[CAL], raw),
        "shape": qf_r + qs_r,
        "comp": out["af"] * qf_r + out["as"] * qs_r,
    }
    for nm, sim in cfgs.items():
        p = panel(sim.loc[VAL], q_obs.loc[VAL])
        out.update({f"{nm}_{k2}": v for k2, v in p.items()})
    obs_p = panel(q_obs.loc[VAL].dropna(), q_obs.loc[VAL].dropna())
    out["obs_acf_integral"] = obs_p.get("acf_integral")
    out["obs_bfi"] = obs_p.get("BFI")
    return out
