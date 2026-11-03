"""Phase 6d.2/6d.3: pack-only, OCV-free resistance GP on the BattGP LFP field data (the 'cheap telemetry' setting).

Pack voltage = sum of the 8 cell voltages; per-cell voltages, balancing currents and the BMS SOC are NOT used. Inputs: pack voltage,
pack current, mean of the four temperature sensors (optional). Observation model (OCV-free): for each discharge run (current < -5 A,
BattGP sign convention: negative = discharge) the reference voltage is the median pack voltage of the rest samples (|I| < 2 A) in the
5 minutes before the run; R_obs = (V_rest - V) / |I|. Resistance = Wiener-velocity time process + SE over the operating point
(|I|, T) with a per-run random offset (same model as the lead-acid work, sunfix.resistance). Causal filter only.

Alarm rules on the causal pack resistance trajectory (no labels): one-sided CUSUM on the standardised resistance (baseline = days 30-60
of the section), or a windowed rule for limited history. Pseudo-labels come from BattGP's per-cell weakest-link probability
(sunfix.lfp_reproduce, paper settings, mean reference operating point). All time axes are days since the start of the analysed section.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .fit_hyper import BOUNDS, neg_log_post, unpack
from .resistance import Normaliser, Prepared, fit_normaliser, run_battery, T_NORM

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "battgp_cache"
RESD = ROOT / "data" / "battgp_results"
PACKD = ROOT / "data" / "battgp_pack"
CELLS = [f"U_Cell_{i}" for i in range(1, 9)]
TEMPS = ["T_Cell_1_2", "T_Cell_3_4", "T_Cell_5_6", "T_Cell_7_8"]
GAP_DAYS = 90.0
BURN, BASE0, BASE1 = 30.0, 30.0, 60.0     # days: filter burn-in, baseline window for the alarm rule
SIGMA_V = 0.05                            # V, pack voltage noise (hyperparameter-free assumption)
I_MIN = 5.0                               # A, observation threshold (rows with I < -5 A are used)
I_RUN = 2.0                               # A, a discharge run is a stretch with I < -2 A (so small current wobble does not split it)
MIN_RUN_S = 60.0
MAX_ROWS_RUN = 40                          # rows per discharge run (evenly thinned): bounds the per-run matrix factorisation of the GP
REST_I = 1.0                              # A, rest = |I| below this


@dataclass(frozen=True)
class Spec:
    name: str = "full"
    interval: float | None = None   # s, keep first sample per bin
    temp: bool = True
    v_round: float | None = None
    i_round: float | None = None
    loss: float = 0.0
    history: int | None = None      # days (windowed alarm rule)


def load_pack(sid: int) -> pd.DataFrame:
    d = pd.read_feather(CACHE / f"{sid}.feather", columns=["Timestamp", "I_Batt"] + CELLS + TEMPS)
    t = (pd.to_datetime(d.Timestamp) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()   # seconds (the feather stores datetime64[us])
    cells = d[CELLS].to_numpy(float)
    ok = np.isfinite(cells).all(1) & (cells > 2.0).all(1) & (cells < 4.3).all(1) & np.isfinite(d.I_Batt.to_numpy()) & (np.abs(d.I_Batt.to_numpy()) < 300)
    out = pd.DataFrame(dict(t=t, V=cells.sum(1), I=d.I_Batt.to_numpy(float), T=d[TEMPS].to_numpy(float).mean(1)))[ok]
    out = out.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    sec = _sections().loc[sid]         # the exact window BattGP's per-cell run analysed (results/lfp_section_times.csv)
    t0, t1 = sec.start.timestamp(), sec.end.timestamp()
    return out[(out.t >= t0) & (out.t <= t1)].reset_index(drop=True)


_SEC = None


def _sections() -> pd.DataFrame:
    global _SEC
    if _SEC is None:
        _SEC = pd.read_csv(ROOT / "results" / "lfp_section_times.csv", parse_dates=["start", "end"]).set_index("system")
    return _SEC


def degrade(d: pd.DataFrame, s: Spec, sid: int) -> pd.DataFrame:
    if s.loss > 0:
        d = d[np.random.default_rng(100 + sid).random(len(d)) >= s.loss]
    if s.interval:
        d = d[~pd.Series((d.t // s.interval).astype("int64").to_numpy()).duplicated().to_numpy()]
    d = d.copy()
    if not s.temp:
        d["T"] = 25.0
    if s.v_round:
        d["V"] = np.round(d.V / s.v_round) * s.v_round
    if s.i_round:
        d["I"] = np.round(d.I / s.i_round) * s.i_round
    return d.reset_index(drop=True)


def section_start(sid: int) -> float:
    return _sections().loc[sid].start.timestamp()


def prepare_pack(d: pd.DataFrame, s: Spec, t_start: float) -> Prepared | None:
    """Discharge runs -> resistance observations with rest-voltage reference (OCV-free)."""
    t, I, V, T = (d[c].to_numpy() for c in ("t", "I", "V", "T"))
    gap_lim = max(180.0, 3.0 * (s.interval or 0.0))
    dis = I < -I_RUN
    if s.temp:
        dis &= (T > 10) & (T < 100)
    rest = np.abs(I) < REST_I
    brk = np.r_[True, (np.diff(t) > gap_lim) | (dis[1:] != dis[:-1])]
    starts = np.flatnonzero(brk)
    stops = np.r_[starts[1:], len(t)]
    rows_t, rows_x, rows_y, rows_n, rows_seg, rows_v, rows_i = [], [], [], [], [], [], []
    seg_id = 0
    bin_s = max(60.0, s.interval or 0.0)
    for a, b in zip(starts, stops):
        if not dis[a] or t[b - 1] - t[a] < max(MIN_RUN_S, 1.5 * (s.interval or 0.0)) or a == 0:
            continue
        lo = int(np.searchsorted(t, t[a] - max(300.0, 2.0 * (s.interval or 0.0)), side="left"))   # rest lookback scales with the sampling interval   # windowed lookup (was a full-array scan per run: quadratic)
        seg_rest = rest[lo:a]
        if not seg_rest.any():
            continue
        v_rest = float(np.median(V[lo:a][seg_rest]))
        idx = np.arange(a, b)
        idx = idx[I[idx] < -I_MIN]
        if len(idx) < (2 if s.interval and s.interval >= 900 else 3):
            continue
        idx = idx[~pd.Series((t[idx] // bin_s).astype("int64")).duplicated().to_numpy()]
        if len(idx) > MAX_ROWS_RUN:
            idx = idx[np.linspace(0, len(idx) - 1, MAX_ROWS_RUN).astype(int)]
        ic = -I[idx]
        r = (v_rest - V[idx]) / ic
        keep = (r > 0) & (r < 0.5)
        idx, ic, r = idx[keep], ic[keep], r[keep]
        if len(idx) < (2 if s.interval and s.interval >= 900 else 3):
            continue
        rows_t.append((t[idx] - t_start) / T_NORM); rows_y.append(r * 1e3); rows_n.append((SIGMA_V / ic) ** 2 * 1e6)   # resistance in mOhm: keeps the GP scale parameters O(1)
        rows_x.append(np.c_[ic, T[idx]] if s.temp else ic[:, None]); rows_seg.append(np.full(len(idx), seg_id)); rows_v.append(V[idx]); rows_i.append(-ic)
        seg_id += 1
    if not rows_t:
        return None
    cat = np.concatenate
    return Prepared(t=cat(rows_t), seg=cat(rows_seg), X=np.vstack(rows_x), y=cat(rows_y), noise=cat(rows_n), v=cat(rows_v), i=cat(rows_i), base_v=np.zeros(len(cat(rows_t))))


def fit_hypers(preps, d_in, seed=0, maxiter=200, max_runs=300):
    """Label-free marginal-likelihood fit on a thinned copy (first max_runs discharge runs of each battery) for speed."""
    preps = [_head(p, max_runs) for p in preps]
    norm = fit_normaliser(preps, n_basis=12, seed=seed)
    d = norm.basis.shape[1]
    theta0 = np.log([3.0, 3.0, 3.0, *([3.0] * d)])   # sf0, sf1, tau in mOhm units
    saved = dict(BOUNDS)                              # the lead-acid bounds (Ohm units) pinned tau at 3; widen for mOhm units, restore afterwards
    BOUNDS.update(sf0=(0.01, 300.0), sf1=(0.01, 300.0), tau=(1e-3, 300.0))
    try:
        res = minimize(neg_log_post, theta0, args=(preps, norm), method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-1))
    finally:
        BOUNDS.update(saved)
    sf0, sf1, tau, ls = unpack(res.x, d)
    return dict(sf0=sf0, sf1=sf1, tau=tau, ls=list(ls), norm_mean=norm.mean.tolist(), norm_std=norm.std.tolist(), basis=norm.basis.tolist(), nlp=float(res.fun))


def _head(p: Prepared, n_runs: int) -> Prepared:
    """First n_runs discharge runs, evenly spread over the whole section (every k-th run)."""
    segs = np.unique(p.seg)
    if len(segs) <= n_runs:
        return p
    keep = segs[np.linspace(0, len(segs) - 1, n_runs).astype(int)]
    m = np.isin(p.seg, keep)
    return Prepared(t=p.t[m], seg=p.seg[m], X=p.X[m], y=p.y[m], noise=p.noise[m], v=p.v[m], i=p.i[m], base_v=p.base_v[m])


def trajectory(prep: Prepared, hyp: dict, t0: float = 0.0):
    norm = Normaliser(np.array(hyp["norm_mean"]), np.array(hyp["norm_std"]), np.array(hyp["basis"]))
    traj, _, _ = run_battery(prep, norm, hyp["sf0"], hyp["sf1"], tuple(hyp["ls"]), tau=hyp["tau"], t0=t0)
    traj["day"] = traj.t * (T_NORM / 86400.0)
    return traj


def alarm_days(traj: pd.DataFrame, hs, k=0.5, history=None):
    """First alarm day for each CUSUM threshold h (nan if none). Standardised against the days 30-60 baseline of the section
    (or, with limited history H, against the first 30 days of the last H days, re-evaluated at each step)."""
    day, r = traj.day.to_numpy(), traj.r0.to_numpy()
    out = {}
    if history is None:
        b0 = max(BASE0, float(day.min()))                  # baseline = 30 days starting at day 30 or the first observation, whichever is later
        b = (day >= b0) & (day <= b0 + 30.0)
        if b.sum() < 3:
            return {h: np.nan for h in hs}
        BASE1_ = b0 + 30.0
        mu = np.median(r[b]); sd = max(1.4826 * np.median(np.abs(r[b] - mu)), 0.02 * abs(mu))   # r is in mOhm
        z = (r - mu) / sd
        S, acc = np.zeros(len(z)), 0.0
        for j in range(len(z)):
            acc = max(0.0, acc + z[j] - k) if day[j] > BASE1_ else 0.0
            S[j] = acc
        for h in hs:
            i = np.flatnonzero(S > h)
            out[h] = float(day[i[0]]) if len(i) else np.nan
        return out
    # limited history: compare with the median of the first 30 days of the visible window; same CUSUM-free windowed z, threshold h
    zs = np.full(len(r), np.nan)
    for j in range(len(r)):
        w = (day > day[j] - history) & (day <= day[j])
        b = w & (day <= day[j] - history + 30.0)
        if b.sum() < 3 or day[j] < history:
            continue
        mu = np.median(r[b]); sd = max(1.4826 * np.median(np.abs(r[b] - mu)), 0.02 * abs(mu))
        zs[j] = (np.median(r[w & (day > day[j] - 14.0)]) - mu) / sd
    for h in hs:
        i = np.flatnonzero(zs > h)
        out[h] = float(day[i[0]]) if len(i) else np.nan
    return out


def cell_labels(sid: int):
    """Pseudo-labels from the per-cell weakest-link probability (paper settings, mean reference): smoothed (acausal) and causal onset days."""
    res = {}
    for k in ("smoothed", "causal"):
        p = RESD / f"fault_probs_sys{sid}_{k}_paper_meanref.parquet"
        if not p.exists():
            return None
        f = pd.read_parquet(p)
        t = f["t"].to_numpy(); w = f["Weakest_link_stat"].to_numpy(float)
        i = np.flatnonzero((w > 0.5) & (t > BURN))
        res[k] = float(t[i[0]]) if len(i) else np.nan
        res[f"span_{k}"] = float(t[-1])
        if k == "smoothed":
            cols = [f"R{c} band_i fault prob" for c in range(1, 9)]
            m = f[cols].to_numpy(float)[t > BURN]
            res["cell"] = int(np.nanargmax(np.nanmax(m, 0))) + 1 if m.size else -1
    return res


def cell_drift_labels(sid: int, band: float = 0.33e-3, sustain_days: float = 7.0, min_span: float = 90.0):
    """Per-cell pseudo-labels from BattGP's per-cell resistance trajectories (paper settings, mean reference operating point).
    Relative resistance of cell i = R_i - mean of the other cells. A fault onset is the first day after the baseline window where, for any
    cell, the relative resistance has risen by at least `band` (the paper's b = 0.33 mOhm) above that cell's own days 30-60 baseline and stays
    there for `sustain_days`. 'acausal' uses the smoothed trajectories (best-knowledge label), 'causal' the filter output (what a per-cell
    online method would know). Systems with a section shorter than `min_span` days are not labelled (no baseline plus room to detect)."""
    p = RESD / f"gp_res_df_sys{sid}_paper_meanref.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    t = d.t.to_numpy()
    out = dict(span=float(t[-1]), eligible=bool(t[-1] >= min_span))
    for kind in ("acausal", "causal"):
        R = np.vstack([d[f"r0_{kind}_c{k}"].to_numpy(float) for k in range(1, 9)])
        rel = R - (R.sum(0, keepdims=True) - R) / 7.0
        b = (t >= BASE0) & (t <= BASE1)
        if not out["eligible"] or b.sum() < 3:
            out[f"onset_{kind}"], out[f"cell_{kind}"] = np.nan, -1
            continue
        drift = rel - np.median(rel[:, b], axis=1, keepdims=True)
        onset, cell = np.nan, -1
        for j in np.flatnonzero(t > BASE1):
            sel = (t >= t[j]) & (t <= t[j] + sustain_days)
            if t[-1] < t[j] + sustain_days:
                break
            hit = (drift[:, sel] >= band).all(axis=1)
            if hit.any():
                onset, cell = float(t[j]), int(np.argmax(drift[:, j])) + 1
                break
        out[f"onset_{kind}"], out[f"cell_{kind}"] = onset, cell
        out[f"max_drift_mohm_{kind}"] = float(np.max(drift[:, t > BASE1]) * 1e3) if (t > BASE1).any() else np.nan
    return out


def rise_days(traj: pd.DataFrame, deltas, sustain: float = 7.0, history: int | None = None):
    """First day on which the causal pack resistance has been at least (1 + delta) x its baseline for `sustain` consecutive days.
    Baseline = median of the 30 days starting at day 30 (or the first observation if later); with limited history H the baseline is the
    first 30 days of the visible window [t - H, t]. Relative thresholds are what an operator would set (e.g. +20%)."""
    day, r = traj.day.to_numpy(), traj.r0.to_numpy()
    out = {}
    if history is None:
        b0 = max(BASE0, float(day.min()))
        b = (day >= b0) & (day <= b0 + 30.0)
        if b.sum() < 3 or np.median(r[b]) <= 0:
            return {d: np.nan for d in deltas}
        ratio = r / np.median(r[b]); ok_from = b0 + 30.0
        for d in deltas:
            hit = (ratio >= 1.0 + d) & (day > ok_from)
            out[d] = _sustained(day, hit, sustain)
        return out
    ratio = np.full(len(r), np.nan)
    for j in range(len(r)):
        if day[j] < history:
            continue
        b = (day > day[j] - history) & (day <= day[j] - history + 30.0)
        if b.sum() < 3 or np.median(r[b]) <= 0:
            continue
        ratio[j] = np.median(r[(day > day[j] - 7.0) & (day <= day[j])]) / np.median(r[b])
    for d in deltas:
        out[d] = _sustained(day, np.nan_to_num(ratio, nan=0.0) >= 1.0 + d, sustain)
    return out


def _sustained(day, hit, sustain):
    i = np.flatnonzero(hit)
    for j in i:
        end = np.searchsorted(day, day[j] + sustain, side="right") - 1
        if day[-1] < day[j] + sustain:
            return np.nan
        if hit[j:end + 1].all():
            return float(day[j])
    return np.nan
