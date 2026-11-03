"""Features at a decision time tau, using only data up to tau (no look-ahead).

Resistance features come from the causal filter state at the last charging segment before tau, extrapolated to tau with
the Wiener-velocity model. Stress features follow Aitio & Howey supplement section 5.
"""
import numpy as np
import pandas as pd

from .clean import clean_raw
from .load import load_meta

HORIZONS = (0, 14, 28, 42, 56)   # days before the end of each series
RES_COLS = ["level", "slope_now", "slope_2w", "slope_4w", "slope_8w", "rel_level", "unc", "stale_days"]
STRESS_COLS = ["age", "ah", "mean_T", "mean_V", "cycles", "float_h"]
RES_CORE = [c for c in RES_COLS if c not in ("unc", "stale_days")]   # no data-availability columns
SETS = {"res": RES_COLS, "stress": STRESS_COLS, "res+stress": RES_COLS + STRESS_COLS, "age_only": ["age"],
        "res_core": RES_CORE, "res_core+stress": RES_CORE + STRESS_COLS, "stale_only": ["stale_days"],
        # Phase 4 sets (registered here so joblib worker processes see them)
        "stress_only": STRESS_COLS, "stress_only_noT": [c for c in STRESS_COLS if c != "mean_T"],
        "res_core+stress_noT": RES_CORE + [c for c in STRESS_COLS if c != "mean_T"],
        "proxy_only": RES_CORE, "proxy+stress": RES_CORE + ["age", "ah", "mean_T", "mean_V", "cycles"], "stress_only_daily": ["age", "ah", "mean_T", "mean_V", "cycles"]}
DT_CAP = 610.0   # s; gaps longer than this contribute no throughput/time


def end_time(batt_id: int, m: pd.DataFrame, d: pd.DataFrame) -> float:
    """End of the series (unix s): repair date for failed batteries, last telemetry for alive ones."""
    return m.failure_time[batt_id].timestamp() if m.failed[batt_id] else float(d.t.iloc[-1])


def _runs(mask: np.ndarray, t: np.ndarray, min_dur: float = 600.0):
    """End times and durations of runs of mask==True (consecutive samples, no gap > DT_CAP) longer than min_dur."""
    brk = np.r_[True, (np.diff(t) > DT_CAP) | (mask[1:] != mask[:-1])]
    start = np.flatnonzero(brk)
    stop = np.r_[start[1:], len(t)] - 1
    keep = mask[start] & ((t[stop] - t[start]) > min_dur)
    return t[stop][keep], (t[stop] - t[start])[keep]


def stress_features(batt_id: int, m: pd.DataFrame | None = None, horizons=HORIZONS, d: pd.DataFrame | None = None,
                    t_end: float | None = None) -> pd.DataFrame:
    """d / t_end: optional degraded frame and native series end (Phase 4); default is native cleaned data."""
    m = m if m is not None else load_meta()
    d = clean_raw(batt_id, m) if d is None else d
    t, I, V, T = (d[c].to_numpy() for c in ("t", "I", "V", "T"))
    dt = np.minimum(np.r_[np.diff(t), 0.0], DT_CAP)
    cum_dt = np.cumsum(dt)
    cum_ah = np.cumsum(np.abs(I) * dt) / 3600.0
    cum_T, cum_V = np.cumsum(T * dt), np.cumsum(V * dt)
    cyc_end, _ = _runs(I > 0.05, t)
    flt_end, flt_dur = _runs((V >= 13.4) & (V <= 13.6), t)
    flt_cum = np.cumsum(flt_dur)
    t_end = end_time(batt_id, m, d) if t_end is None else t_end
    act = m.activation_time[batt_id].timestamp()
    rows = []
    for h in horizons:
        tau = t_end - h * 86400.0
        k = np.searchsorted(t, tau, side="right") - 1
        if k < 1:
            rows.append(dict(ID=batt_id, h=h, tau=tau)); continue
        nf = np.searchsorted(flt_end, tau, side="right")
        rows.append(dict(ID=batt_id, h=h, tau=tau, age=(tau - act) / 86400.0, ah=cum_ah[k], mean_T=cum_T[k] / cum_dt[k],
                         mean_V=cum_V[k] / cum_dt[k], cycles=int(np.searchsorted(cyc_end, tau, side="right")),
                         float_h=(flt_cum[nf - 1] / 3600.0 if nf else 0.0)))
    return pd.DataFrame(rows)


def resistance_features(batt_id: int, traj: pd.DataFrame, taus_days: dict[int, float], window_baseline: bool = False) -> pd.DataFrame:
    """traj: causal per-segment filter output (sunfix.resistance.run_battery); taus_days: h -> tau in days since activation."""
    td = traj.t.to_numpy() * 400.0
    base_rows = (td >= 60) & (td <= 120)
    if window_baseline and len(td):   # Phase 4 short history: days 60-120 are not visible; use the first quarter of the visible span
        base_rows = td <= td[0] + 0.25 * max(td[-1] - td[0], 1.0)
    base = float(np.median(traj.r0[base_rows])) if base_rows.any() else np.nan
    rows = []
    for h, tau in taus_days.items():
        k = np.searchsorted(td, tau, side="right") - 1
        if k < 0:
            rows.append(dict(ID=batt_id, h=h)); continue
        r = traj.iloc[k]
        dlt = (tau - td[k]) / 400.0
        level = float(r.r_op + r.z0 + r.z1 * dlt)
        var = float(r.r0var + dlt**2 * r.p11 + 2 * dlt * r.p01)
        out = dict(ID=batt_id, h=h, level=level, slope_now=float(r.z1) / 400.0 * 1000.0, rel_level=level / base if base == base else np.nan,
                   unc=np.sqrt(max(var, 0.0)), stale_days=tau - td[k])
        for w, name in ((14, "slope_2w"), (28, "slope_4w"), (56, "slope_8w")):
            sel = (td <= tau) & (td >= tau - w)
            out[name] = float(np.polyfit(td[sel], traj.r0.to_numpy()[sel], 1)[0] * 1000.0) if sel.sum() >= 3 else out["slope_now"]
        rows.append(out)
    return pd.DataFrame(rows)
