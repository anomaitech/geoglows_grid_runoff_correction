"""Observation-inverted runoff for all 37 CAMELS basins (historical, no ML).

Per basin, using its own locally identified operators (Tf, Ts, phi from
mb_results.csv) and routing k:
    Q_obs -> deconvolve routing -> Lyne-Hollick separation
          -> invert fast buffer (exact, well-conditioned)
          -> invert slow operator (Tikhonov-regularized NNLS; smoothness is
             the identifiability limit of the store, not a choice)
Product: data/mb/inverted/{gid}.csv  (Rfast_mm, Rslow_mm, mm/day)
Integrity per basin: closure r (re-routed recombination vs Q_obs) and the
independent-precipitation test r(R_fast, Daymet) vs r(model sro, Daymet).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).parent))
from config import PROJECT_DIR, RESULTS
from mb_core import lyne_hollick

MB = PROJECT_DIR / "data" / "mb"
OUT = MB / "inverted"
OUT.mkdir(exist_ok=True)
LAM = 0.3


def kernel(T: float, n: int) -> np.ndarray:
    if T <= 0:
        k = np.zeros(n); k[0] = 1.0
        return k
    a = np.exp(-1 / T)
    k = np.empty(n)
    k[0] = 1 - T * (1 - a)
    rem = 1 - k[0]
    for i in range(1, n):
        k[i] = rem * (1 - a)
        rem *= a
    return k


def conv_matrix(kern, n, trunc=1e-6):
    m = int(np.searchsorted(np.cumsum(kern), 1 - trunc)) + 1
    diags = [np.full(n - i, kern[i]) for i in range(min(m, n))]
    return sparse.diags(diags, offsets=[-i for i in range(min(m, n))],
                        format="csr")


def invert(y, A, lam, iters=3000):
    n = len(y)
    D = sparse.diags([np.ones(n - 1), -np.ones(n - 1)], [0, 1],
                     shape=(n - 1, n), format="csr")
    Aa = sparse.vstack([A, lam * D])
    x = lsqr(Aa, np.concatenate([y, np.zeros(n - 1)]), iter_lim=iters)[0]
    return np.clip(x, 0, None)


if __name__ == "__main__":
    res = pd.read_csv(RESULTS / "mb_results.csv", dtype={"gauge_id": str}
                      ).set_index("gauge_id")
    rids = json.load(open(MB / "river_ids.json"))
    rows = []
    for gid, b in res.iterrows():
        if not np.isfinite(b.get("Tf", np.nan)):
            continue
        q = pd.read_csv(MB / "qobs" / f"{gid}.csv", index_col=0,
                        parse_dates=True)["q_cms"]
        q = q.interpolate(limit=7).dropna()
        cells = json.loads((MB / "weights" / f"{gid}.json").read_text())
        area = sum(c["inter_km2"] for c in cells)
        to_mm = 86400 / (area * 1e6) * 1000
        k_route = rids.get(gid, {}).get("musk_k_s", 21600) / 86400
        n = len(q)
        if n < 3000:
            continue
        # 1) routing deconvolution
        Ar = conv_matrix(kernel(k_route, n), n)
        lat = invert(q.to_numpy(), Ar, lam=0.05) * to_mm
        # 2) separation
        slow_lat = lyne_hollick(pd.Series(lat, q.index))
        fast_lat = np.clip(lat - slow_lat.to_numpy(), 0, None)
        # 3) fast inversion
        Af = conv_matrix(kernel(b.Tf, n), n)
        r_fast = invert(fast_lat, Af, lam=0.1)
        # 4) slow inversion
        As = ((1 - b.phi) * sparse.identity(n, format="csr")
              + b.phi * conv_matrix(kernel(b.Ts, n), n))
        r_slow = invert(slow_lat.to_numpy(), As, lam=LAM)
        pd.DataFrame({"Rfast_mm": r_fast, "Rslow_mm": r_slow},
                     index=q.index).round(5).to_csv(OUT / f"{gid}.csv")
        # integrity checks
        recon = Ar @ (Af @ r_fast + As @ r_slow) / to_mm
        closure = float(np.corrcoef(recon, q)[0, 1])
        row = {"gauge_id": gid, "closure_r": closure,
               "Rfast_mm_yr": r_fast.mean() * 365.25,
               "Rslow_mm_yr": r_slow.mean() * 365.25,
               "fast_share": r_fast.sum() / (r_fast.sum() + r_slow.sum())}
        # independent precip test + model comparison
        try:
            dm = pd.read_csv(MB / "daymet" / f"{gid}.csv", index_col=0,
                             parse_dates=True)["prcp_mm"].reindex(q.index)
            f = pd.read_csv(MB / "era5" / f"{gid}.csv", index_col=0,
                            parse_dates=True).reindex(q.index)
            m = dm.notna() & f.sro_mm.notna()
            row["r_Rfast_P"] = float(np.corrcoef(r_fast[m], dm[m])[0, 1])
            row["r_sro_P"] = float(np.corrcoef(f.sro_mm[m], dm[m])[0, 1])
            row["era5_sro_mm_yr"] = float(f.sro_mm.mean() * 365.25)
            row["era5_ssro_mm_yr"] = float(f.ssro_mm.mean() * 365.25)
        except Exception:
            pass
        rows.append(row)
        print(f"{gid}: closure={closure:.3f} "
              f"Rf={row['Rfast_mm_yr']:.0f} Rs={row['Rslow_mm_yr']:.0f} "
              f"r(Rf,P)={row.get('r_Rfast_P', float('nan')):.2f} "
              f"vs r(sro,P)={row.get('r_sro_P', float('nan')):.2f}",
              flush=True)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(RESULTS / "mb_inverted_summary.csv", index=False)
    print(f"\n=== summary over {len(t)} basins ===")
    print(f"closure r: median {t.closure_r.median():.3f} "
          f"min {t.closure_r.min():.3f}")
    print(f"precip test: r(Rfast,P) median {t.r_Rfast_P.median():.2f} vs "
          f"r(sro,P) median {t.r_sro_P.median():.2f}; "
          f"inversion wins at {(t.r_Rfast_P > t.r_sro_P).sum()}/{len(t)}")
    print(f"fast share: median {t.fast_share.median():.2f} "
          f"(ERA5-Land: {(t.era5_sro_mm_yr/(t.era5_sro_mm_yr+t.era5_ssro_mm_yr)).median():.2f})")
