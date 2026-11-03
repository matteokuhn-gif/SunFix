"""Phase 4: minimum-data experiments. Degrade the telemetry, re-run segmentation, resistance filter and classifiers.

Fixed across scenarios (so numbers are comparable): batteries, decision times (defined from NATIVE data), the 5 CV folds,
classifiers, thresholds. Per-fold GP hyperparameters are those fitted on native data in Phase 3 (a degraded-data refit could do
slightly better; documented limitation), except `no_temperature`, which has one fewer input and is refitted per fold.

Usage: python -m sunfix.phase4 <scenario> [<scenario> ...] | list | all | table
"""
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .classify import metrics, one
from .features import SETS
from .clean import SEG, charging_segments, clean_raw, interpolate_segment
from .features import RES_CORE, STRESS_COLS, end_time, resistance_features, stress_features
from .fit_hyper import fit
from .load import ROOT, load_meta, zip_index
from .phase3_features import N_FOLDS, folds
from .resistance import Normaliser, prepare, run_battery
from .soc import add_soc

HS = (28, 56)               # decision horizons in days (4 and 8 weeks), the two the brief asks for
FRAC = float(os.environ.get("SUNFIX_SUB_FRAC", "1.0"))   # fraction of batteries (stratified, seed 0); 1.0 = all
OUT = ROOT / "data" / ("phase4" if FRAC >= 1 else f"phase4_sub{int(FRAC * 100)}")


def subsample(m: pd.DataFrame, ids: list[int]) -> set[int]:
    """Fixed stratified subsample (same batteries in every scenario)."""
    if FRAC >= 1:
        return set(ids)
    rng = np.random.default_rng(0)
    keep = []
    for flag in (True, False):
        pool = np.array(sorted(i for i in ids if m.failed[i] == flag))
        keep += rng.choice(pool, int(round(FRAC * len(pool))), replace=False).tolist()
    return set(keep)
RES_DIR = ROOT / "results"


@dataclass(frozen=True)
class Spec:
    group: str
    interval: float | None = None   # s; keep the first sample in each bin of this length
    temp: bool = True
    history: int | None = None      # days of history visible at decision time
    v_round: float | None = None    # V
    i_round: float | None = None    # A
    loss: float = 0.0               # fraction of rows dropped at random
    daily: bool = False             # daily summaries only (no sub-daily data)


SCENARIOS: dict[str, Spec] = {
    "native_1min": Spec("sampling interval"),
    "interval_5min": Spec("sampling interval", interval=300.0),
    "interval_15min": Spec("sampling interval", interval=900.0),
    "interval_1h": Spec("sampling interval", interval=3600.0),
    "daily_summary": Spec("sampling interval", daily=True),
    "no_temperature": Spec("temperature", temp=False),
    "history_30d": Spec("history length", history=30),
    "history_90d": Spec("history length", history=90),
    "history_180d": Spec("history length", history=180),
    "round_v0.1": Spec("precision", v_round=0.1),
    "round_v0.01": Spec("precision", v_round=0.01),
    "round_i0.1": Spec("precision", i_round=0.1),
    "loss_10": Spec("packet loss", loss=0.10),
    "loss_30": Spec("packet loss", loss=0.30),
    "loss_50": Spec("packet loss", loss=0.50),
}


def degrade(d: pd.DataFrame, s: Spec, batt_id: int) -> pd.DataFrame:
    """Apply a degradation to a cleaned frame (t, I, V, T). Order: packet loss, thinning, temperature, rounding."""
    if s.loss > 0:
        keep = np.random.default_rng(1000 + batt_id).random(len(d)) >= s.loss
        d = d[keep]
    if s.interval:
        d = d[~pd.Series((d.t // s.interval).astype("int64").to_numpy()).duplicated().to_numpy()]   # first sample per bin
    d = d.copy()
    if not s.temp:
        d["T"] = 25.0
    if s.v_round:
        d["V"] = np.round(d.V / s.v_round) * s.v_round
    if s.i_round:
        d["I"] = np.round(d.I / s.i_round) * s.i_round
    return d.reset_index(drop=True)


def seg_criteria(s: Spec) -> dict:
    c = dict(SEG)
    if s.interval:
        c["max_gap"] = max(SEG["max_gap"], 2.5 * s.interval)   # a thinned stream must not be broken by its own sampling period
    return c


def stride_for(s: Spec) -> int:
    return max(5, int(round((s.interval or 0) / 60.0)))       # about one resistance observation per real sample


# ---------------------------------------------------------------- daily-summary scenario (no resistance GP possible)
def daily_features(i: int, m: pd.DataFrame, taus: dict[int, float]) -> pd.DataFrame:
    """Features from daily min/max/mean of V, I, T plus Wh in/out only. Resistance proxy: (Vmax-Vmin)/(Imax-Imin), a crude
    'voltage swing per amp swing' that mixes resistance with state-of-charge change."""
    d = clean_raw(i, m)
    day = (d.t // 86400).astype("int64")
    dt = np.minimum(np.r_[np.diff(d.t.to_numpy()), 0.0], 610.0)
    w = (d.V * d.I * dt / 3600.0)
    g = pd.DataFrame(dict(day=day, V=d.V, I=d.I, T=d["T"], whin=np.where(d.I < 0, -w, 0.0), whout=np.where(d.I > 0, w, 0.0), dt=dt))
    a = g.groupby("day").agg(vmin=("V", "min"), vmax=("V", "max"), vmean=("V", "mean"), imin=("I", "min"), imax=("I", "max"),
                             tmean=("T", "mean"), whin=("whin", "sum"), whout=("whout", "sum"), hours=("dt", lambda x: x.sum() / 3600.0))
    a["proxy"] = np.where((a.imax - a.imin) > 1.0, (a.vmax - a.vmin) / (a.imax - a.imin), np.nan)
    act = m.activation_time[i].timestamp() / 86400.0
    dd = a.index.to_numpy() - act
    rows = []
    for h, tau in taus.items():
        k = dd <= tau
        if k.sum() < 5:
            rows.append(dict(ID=i, h=h)); continue
        x, p = dd[k], a.proxy.to_numpy()[k]
        out = dict(ID=i, h=h, age=tau, ah=float((a.whin.to_numpy()[k] + a.whout.to_numpy()[k]).sum() / a.vmean.to_numpy()[k].mean()),
                   mean_T=float(a.tmean.to_numpy()[k].mean()), mean_V=float(a.vmean.to_numpy()[k].mean()),
                   cycles=int((a.whout.to_numpy()[k] > 0.5).sum()))
        ok = ~np.isnan(p)
        base = p[ok & (x >= 60) & (x <= 120)]
        lv = p[ok & (x >= tau - 14)]
        out["level"] = float(np.median(lv)) if len(lv) else np.nan
        out["rel_level"] = out["level"] / float(np.median(base)) if len(base) and len(lv) else np.nan
        for wk, name in ((14, "slope_2w"), (28, "slope_4w"), (56, "slope_8w")):
            sel = ok & (x >= tau - wk)
            out[name] = float(np.polyfit(x[sel], p[sel], 1)[0] * 1000.0) if sel.sum() >= 3 else np.nan
        out["slope_now"] = out["slope_2w"]
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-battery feature computation
def one_battery(i, m, s: Spec, hypers, variant):
    """Returns (stress rows, resistance rows for every fold, samples per day after degradation)."""
    d0 = clean_raw(i, m)
    t_end = end_time(i, m, d0)
    act = m.activation_time[i].timestamp()
    taus = {h: (t_end - h * 86400.0 - act) / 86400.0 for h in HS}
    if s.daily:
        f = daily_features(i, m, taus)
        return f, None, float("nan")
    d = degrade(d0, s, i)
    span_days = max((d.t.iloc[-1] - d.t.iloc[0]) / 86400.0, 1.0) if len(d) else 1.0
    crit, stride = seg_criteria(s), stride_for(s)
    stress, res = [], []
    # one pass over the full degraded series, or one pass per horizon when history is limited
    passes = [(HS, d)] if not s.history else [((h,), d[(d.t >= t_end - h * 86400.0 - s.history * 86400.0) & (d.t <= t_end - h * 86400.0)].reset_index(drop=True)) for h in HS]
    for hs, dd in passes:
        stress.append(stress_features(i, m, horizons=hs, d=dd, t_end=t_end))
        parts = []
        for k, (a, b) in enumerate(charging_segments(dd, crit)):
            q = interpolate_segment(dd, a, b)
            if len(q):
                parts.append(q.assign(seg=k))
        seg = pd.concat(parts, ignore_index=True) if parts else None
        p = prepare(i, variant, m, stride=stride, seg=seg) if seg is not None else None
        for kf, hp in enumerate(hypers):
            if p is None:
                res.append(pd.DataFrame([dict(ID=i, h=h, fold=kf) for h in hs])); continue
            norm = Normaliser(np.array(hp["norm_mean"]), np.array(hp["norm_std"]), np.array(hp["basis"]))
            t0 = float(p.t[0]) if s.history else 0.0
            traj, _, _ = run_battery(p, norm, hp["sf0"], hp["sf1"], tuple(hp["ls"]), tau=hp["tau"], t0=t0)
            r = resistance_features(i, traj, {h: taus[h] for h in hs}, window_baseline=bool(s.history))
            res.append(r.assign(fold=kf))
    return pd.concat(stress, ignore_index=True), pd.concat(res, ignore_index=True), len(d) / span_days


def get_hypers(name: str, s: Spec, tr_folds):
    if s.temp:
        return [json.load(open(ROOT / "data" / f"hyper_f{k}.json")) for k in range(N_FOLDS)], "ocv"
    out = []
    for k, tr in enumerate(tr_folds):
        path = OUT / f"hyper_{name}_f{k}.json"
        out.append(json.load(open(path)) if path.exists() else fit("ocv_noT", pool=tr.tolist(), out_path=path, quiet=True, seed=k))
    return out, "ocv_noT"


def classify(stress, res, s: Spec, m, fl):
    """Same classifier setup as Phase 3 (GPC and logistic regression, threshold 0.5), pooled out-of-fold, h in HS."""
    if s.daily:   # no float-hours channel in daily summaries
        sets = {"stress_only_daily": SETS["stress_only_daily"], "age_only": ["age"], "proxy_only": RES_CORE, "proxy+stress": SETS["proxy+stress"]}
    elif not s.temp:
        sets = {"stress_only_noT": SETS["stress_only_noT"], "age_only": ["age"], "res_core": RES_CORE, "res_core+stress_noT": SETS["res_core+stress_noT"]}
    else:
        sets = {"stress_only": SETS["stress_only"], "age_only": ["age"], "res_core": RES_CORE, "res_core+stress": SETS["res_core+stress"]}
    y_all = m.failed
    jobs = []
    for k, (tr, te) in enumerate(fl):
        for h in HS:
            tab = stress[stress.h == h].set_index("ID")
            if not s.daily:
                r = res[(res.fold == k) & (res.h == h)].set_index("ID").drop(columns=["h", "fold"])
                tab = tab.join(r, how="outer")
            tab = tab.reindex(sorted(set(tr) | set(te)))
            for c in {c for cols in sets.values() for c in cols}:
                if c not in tab:
                    tab[c] = np.nan
            yy = pd.Series(y_all.reindex(tab.index).to_numpy(), index=tab.index)
            for name in sets:
                for mod in ("gpc", "lr"):
                    if mod == "gpc" and name == "age_only":
                        continue
                    jobs.append((h, k, name, mod, tr, te, tab, yy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = Parallel(n_jobs=8)(delayed(one)(h, k, nm, mod, tr, te, tab, yy) for h, k, nm, mod, tr, te, tab, yy in jobs)
    pred = pd.concat([pd.DataFrame(dict(h=o["h"], fold=o["fold"], set=o["set"], model=o["model"], ID=o["ID"], p=o["p"])) for o in out])
    pred["failed"] = y_all.reindex(pred.ID).to_numpy()
    rows = []
    for (h, st, mod), g in pred.groupby(["h", "set", "model"]):
        r = metrics(g.failed.to_numpy(bool), g.p.to_numpy()); r.update(h=h, set=st, model=mod); rows.append(r)
    return pd.DataFrame(rows)


def run(name: str):
    s = SCENARIOS[name]
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_meta(); ids = sorted(zip_index()); fl = folds(m)
    sub = subsample(m, ids)
    ids = [i for i in ids if i in sub]
    fl = [(np.array([i for i in tr if i in sub]), np.array([i for i in te if i in sub])) for tr, te in fl]   # Phase 3 folds, restricted
    t = time.time()
    hypers, variant = get_hypers(name, s, [tr for tr, _ in fl])
    print(f"[{name}] hyperparameters ready {time.time() - t:.0f}s", flush=True)
    out = Parallel(n_jobs=8)(delayed(one_battery)(i, m, s, hypers, variant) for i in ids)
    stress = pd.concat([o[0] for o in out], ignore_index=True)
    res = stress if s.daily else pd.concat([o[1] for o in out], ignore_index=True)
    rate = float(np.nanmedian([o[2] for o in out])) if not s.daily else 2.0
    print(f"[{name}] features {time.time() - t:.0f}s", flush=True)
    r = classify(stress, res, s, m, fl)
    key = "level"
    cov = res.dropna(subset=[key]).groupby("h").size() / (len(ids) * (1 if s.daily else N_FOLDS))
    r["coverage"] = r.h.map(cov)   # share of batteries with a resistance reading at decision time (rest are median-imputed)
    for k_, v in dict(scenario=name, group=s.group, samples_per_day=rate, seconds=time.time() - t).items():
        r[k_] = v
    r.to_csv(OUT / f"{name}.csv", index=False)
    print(r[(r.model == "gpc") | (r.set == "age_only")][["h", "set", "model", "balanced_acc", "coverage"]].round(3).to_string(index=False), flush=True)
    print(f"[{name}] done {time.time() - t:.0f}s", flush=True)


def native_row():
    assert FRAC >= 1, "with a subsample the native reference is the native_1min scenario itself"
    """Native (1-min) reference from the Phase 3 predictions, h in HS, same sets."""
    r = pd.read_csv(RES_DIR / "phase3_results.csv")
    r = r[r.h.isin(HS) & r.set.isin(["res_core", "res_core+stress", "stress", "age_only"]) & r.model.isin(["gpc", "lr"])].copy()
    r["set"] = r["set"].replace({"stress": "stress_only"})
    r["scenario"], r["group"], r["samples_per_day"], r["coverage"] = "native_1min", "sampling interval", 1440.0, np.nan
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["list"]:
        print("\n".join(SCENARIOS))
    elif a[:1] == ["table"]:
        from .phase4_report import build
        build()
    else:
        for n in (list(SCENARIOS) if a == ["all"] else a):
            if not (OUT / f"{n}.csv").exists():
                run(n)
