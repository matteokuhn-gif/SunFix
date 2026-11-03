"""Phase 6b: extended feature set for every system-day (landmark), strictly no look-ahead.

Landmark day d means the clock time activation + d days; every feature uses telemetry strictly before that moment
(complete days < d, resistance-trajectory points with time <= d). Groups:
  res   : GP resistance level, slopes over 7/30/90 d, curvature (90 d), CUSUM statistic, fleet-relative z-score
  usage : equivalent full cycles, partial-charge Ah in the 12-13 V band, end-of-discharge V at matched load (0.5-1.5 A),
          hours at low voltage while discharging, low-voltage-disconnect trips, hours above 35 / 40 degC
  avail : data coverage, outage days and gap hours (+ resistance variance). These can leak (failed batteries' telemetry may stop before
          the repair date), so they are an ablation group and excluded from headline models.
Fleet z-score: level compared with batteries of the same age installed in the 60 days BEFORE this battery (causal; the brief's
'+-1 month' peer window would use peers' later data, so only earlier or equal installs are used).
Output: data/landmarks6.parquet
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .clean import clean_raw
from .features import end_time
from .load import ROOT, load_meta, zip_index
from .landmarks import MIN_AGE
from .phase3_features import folds

CAP_AH = 20.0
DT_CAP = 610.0


def _win(cs, d, w):
    """Sum over complete days [d-w, d-1] from a cumulative array cs (cs[j] = sum of days < j)."""
    return cs[d] - cs[np.maximum(d - w, 0)]


def daily_block(i, m, days):
    d = clean_raw(i, m)
    act = m.activation_time[i].timestamp()
    t, I, V, T = (d[c].to_numpy() for c in ("t", "I", "V", "T"))
    dt = np.minimum(np.r_[np.diff(t), 0.0], DT_CAP)
    di = np.floor((t - act) / 86400.0).astype(int)
    n = int(days.max()) + 2
    ok = (di >= 0) & (di < n)
    def agg(x):
        return np.bincount(di[ok], weights=x[ok], minlength=n)[:n]
    ah_dis = agg(np.maximum(I, 0) * dt) / 3600.0
    ah_band = agg(np.where((V >= 12.0) & (V < 13.0), np.maximum(-I, 0) * dt, 0.0)) / 3600.0
    low_h = agg(np.where((V < 12.0) & (I > 0.05), dt, 0.0)) / 3600.0
    prev = np.r_[V[0], V[:-1]]
    lvd = agg(((V < 10.8) & (prev >= 10.8)).astype(float))
    hot35 = agg(np.where(T > 35, dt, 0.0)) / 3600.0
    hot40 = agg(np.where(T > 40, dt, 0.0)) / 3600.0
    cover = np.minimum(agg(dt) / 86400.0, 1.0)
    # end-of-discharge V at matched load: daily minimum V while discharging at 0.5-1.5 A
    match = ok & (I >= 0.5) & (I <= 1.5)
    eod = np.full(n, np.nan)
    if match.any():
        s = pd.Series(V[match]).groupby(di[match]).min()
        eod[s.index.to_numpy()] = s.to_numpy()
    cs = lambda x: np.r_[0.0, np.cumsum(x)]
    dd = days.astype(int)
    out = pd.DataFrame(dict(
        ID=i, day=days, efc=cs(ah_dis)[dd] / CAP_AH, ah_band_30=_win(cs(ah_band), dd, 30), low_h_30=_win(cs(low_h), dd, 30),
        lvd_30=_win(cs(lvd), dd, 30), hot35_30=_win(cs(hot35), dd, 30), hot40_30=_win(cs(hot40), dd, 30),
        cover_30=_win(cs(cover), dd, 30) / 30.0, outage_days_30=_win(cs((cover < 0.5).astype(float)), dd, 30),
        gap_h_30=_win(cs((1 - cover) * 24.0), dd, 30)))
    ev = pd.DataFrame({"e": eod}).rolling(30, min_periods=3).median().shift(1).to_numpy().ravel()   # shift: complete days before d
    out["eod_v_30"] = ev[dd]
    return out


def traj_block(i, k, days):
    tr = pd.read_parquet(ROOT / "data" / f"resistance_f{k}" / f"{i}.parquet")
    td, r0, var = tr.t.to_numpy() * 400.0, tr.r0.to_numpy(), tr.r0var.to_numpy()
    base = (td >= 60) & (td <= 120)
    mu = np.median(r0[base]) if base.any() else np.nan
    sd = max(1.4826 * np.median(np.abs(r0[base] - mu)), 0.02) if base.any() else np.nan
    x = (r0 - mu) / sd if base.any() else np.full(len(r0), np.nan)
    S = np.zeros(len(x)); acc = 0.0
    for j, v in enumerate(x):
        acc = max(0.0, acc + (0.0 if v != v else v) - 0.5); S[j] = acc      # one-sided CUSUM, allowance 0.5 sd
    cols = {k_: np.full(len(days), np.nan) for k_ in ("slope7", "slope30", "slope90", "curv90", "cusum", "res_var")}
    for n_, d in enumerate(days):
        k_ = np.searchsorted(td, d, side="right") - 1
        if k_ < 0:
            continue
        cols["cusum"][n_], cols["res_var"][n_] = S[k_], var[k_]
        for w, name in ((7, "slope7"), (30, "slope30"), (90, "slope90")):
            sel = (td <= d) & (td >= d - w)
            if sel.sum() >= 3:
                cols[name][n_] = np.polyfit(td[sel], r0[sel], 1)[0] * 1000.0
        sel = (td <= d) & (td >= d - 90)
        if sel.sum() >= 6:
            cols["curv90"][n_] = 2 * np.polyfit(td[sel] - d, r0[sel], 2)[0] * 1e6
    return pd.DataFrame(dict(ID=i, day=days, **cols))


def one(i, m, k, days):
    a = daily_block(i, m, days)
    b = traj_block(i, k, days)
    return a.merge(b, on=["ID", "day"])


def fleet_z(df, window=60):
    """Causal peer z-score of `level` at the same age: peers installed within [own activation - window d, own activation)."""
    b = df.drop_duplicates("ID").set_index("ID")["activation"].sort_values()
    ids, act = b.index.to_numpy(), b.to_numpy() / 86400.0
    mask = (act[None, :] < act[:, None]) & (act[None, :] >= act[:, None] - window)    # row: battery, col: peer
    piv = df.pivot(index="day", columns="ID", values="level")[ids]
    z = pd.DataFrame(np.nan, index=piv.index, columns=ids)
    M = mask.astype(float)
    for d, row in piv.iterrows():
        v = row.to_numpy(); ok = ~np.isnan(v); vv = np.where(ok, v, 0.0)
        cnt = M @ ok.astype(float)
        mean = (M @ vv) / np.maximum(cnt, 1)
        var = (M @ (vv**2)) / np.maximum(cnt, 1) - mean**2
        sdv = np.sqrt(np.maximum(var, 1e-12))
        z.loc[d] = np.where((cnt >= 5) & ok, (v - mean) / np.maximum(sdv, 0.02), np.nan)
    z.index.name, z.columns.name = "day", "ID"
    return z.stack().rename("fleet_z").reset_index()


def main():
    m = load_meta()
    fold_of = {i: k for k, (_, te) in enumerate(folds(m)) for i in te}
    df = pd.read_parquet(ROOT / "data" / "landmarks.parquet")
    groups = {i: g.day.to_numpy() for i, g in df.groupby("ID")}
    parts = Parallel(n_jobs=8)(delayed(one)(i, m, fold_of[i], groups[i]) for i in sorted(groups))
    new = pd.concat(parts, ignore_index=True)
    out = df.merge(new, on=["ID", "day"], how="left")
    z = fleet_z(out)
    out = out.merge(z, on=["ID", "day"], how="left")
    out.to_parquet(ROOT / "data" / "landmarks6.parquet")
    print(len(out), "rows;", out.isna().mean().round(3)[["eod_v_30", "slope7", "slope90", "curv90", "cusum", "fleet_z", "cover_30"]].to_dict())


if __name__ == "__main__":
    main()
