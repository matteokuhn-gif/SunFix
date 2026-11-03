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
    gap = np.flatnonzero(np.diff(out.t.to_numpy()) > GAP_DAYS * 86400)
    if len(gap):                       # keep the latest section without a gap larger than 90 days (as BattGP / the paper)
        out = out.iloc[gap[-1] + 1:].reset_index(drop=True)
    return out


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


def prepare_pack(d: pd.DataFrame, s: Spec, t_start: float) -> Prepared | None:
    """Discharge runs -> resistance observations with rest-voltage reference (OCV-free)."""
    t, I, V, T = (d[c].to_numpy() for c in ("t", "I", "V", "T"))
    gap_lim = max(180.0, 3.0 * (s.interval or 0.0))
    dis = I < -I_RUN
    if s.temp:
        dis &= (T > 10) & (T < 100)
    brk = np.r_[True, (np.diff(t) > gap_lim) | (dis[1:] != dis[:-1])]
    starts = np.flatnonzero(brk)
    stops = np.r_[starts[1:], len(t)]
    rows_t, rows_x, rows_y, rows_n, rows_seg, rows_v, rows_i = [], [], [], [], [], [], []
    seg_id = 0
    bin_s = max(60.0, s.interval or 0.0)
    for a, b in zip(starts, stops):
        if not dis[a] or t[b - 1] - t[a] < MIN_RUN_S or a == 0:
            continue
        pre = (t >= t[a] - 300.0) & (t < t[a]) & (np.abs(I) < REST_I)
        if pre.sum() < 1:
            continue
        v_rest = float(np.median(V[pre]))
        idx = np.arange(a, b)
        idx = idx[I[idx] < -I_MIN]
        if len(idx) < 3:
            continue
        idx = idx[~pd.Series((t[idx] // bin_s).astype("int64")).duplicated().to_numpy()]
        ic = -I[idx]
        r = (v_rest - V[idx]) / ic
        keep = (r > 0) & (r < 0.5)
        idx, ic, r = idx[keep], ic[keep], r[keep]
        if len(idx) < 3:
            continue
        rows_t.append((t[idx] - t_start) / 86400.0 / (T_NORM / 86400.0)); rows_y.append(r); rows_n.append(SIGMA_V**2 / ic**2)
        rows_x.append(np.c_[ic, T[idx]] if s.temp else ic[:, None]); rows_seg.append(np.full(len(idx), seg_id)); rows_v.append(V[idx]); rows_i.append(-ic)
        seg_id += 1
    if not rows_t:
        return None
    cat = np.concatenate
    return Prepared(t=cat(rows_t), seg=cat(rows_seg), X=np.vstack(rows_x), y=cat(rows_y), noise=cat(rows_n), v=cat(rows_v), i=cat(rows_i), base_v=np.zeros(len(cat(rows_t))))


def fit_hypers(preps, d_in, seed=0, maxiter=250):
    norm = fit_normaliser(preps, n_basis=12, seed=seed)
    d = norm.basis.shape[1]
    theta0 = np.log([1.0, 0.5, 0.01, *([3.0] * d)])
    res = minimize(neg_log_post, theta0, args=(preps, norm), method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-1))
    sf0, sf1, tau, ls = unpack(res.x, d)
    return dict(sf0=sf0, sf1=sf1, tau=tau, ls=list(ls), norm_mean=norm.mean.tolist(), norm_std=norm.std.tolist(), basis=norm.basis.tolist(), nlp=float(res.fun))


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
        b = (day >= BASE0) & (day <= BASE1)
        if b.sum() < 3:
            return {h: np.nan for h in hs}
        mu = np.median(r[b]); sd = max(1.4826 * np.median(np.abs(r[b] - mu)), 0.02 * abs(mu))
        z = (r - mu) / sd
        S, acc = np.zeros(len(z)), 0.0
        for j in range(len(z)):
            acc = max(0.0, acc + z[j] - k) if day[j] > BASE1 else 0.0
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
