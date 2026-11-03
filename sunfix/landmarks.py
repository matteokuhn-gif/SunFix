"""Phase 6a: landmark table over ALL system-days (age >= 120 d), features as of each day (no look-ahead), labels 'fails within H days'.

Resistance features come from the out-of-fold causal trajectories of Phase 3 (each battery uses the fold in which it was a test battery).
Alive batteries' last H days have unknown outcomes (right-censored) and are dropped for that horizon; failed batteries' landmarks are
kept up to the repair date. Output: data/landmarks.parquet.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .clean import clean_raw
from .features import RES_CORE, STRESS_COLS, end_time, resistance_features, stress_features
from .load import ROOT, load_meta, zip_index
from .phase3_features import folds

MIN_AGE = 120   # days; the days 60-120 baseline of rel_level is then defined
HS = (30, 56)


def one(i, m, k):
    d = clean_raw(i, m)
    t_end, act = end_time(i, m, d), m.activation_time[i].timestamp()
    end_day = (t_end - act) / 86400.0
    days = np.arange(MIN_AGE, int(np.floor(end_day)) + 1)
    if len(days) == 0:
        return None
    st = stress_features(i, m, horizons=tuple((end_day - days) ), d=d, t_end=t_end)
    st["day"] = days
    traj = pd.read_parquet(ROOT / "data" / f"resistance_f{k}" / f"{i}.parquet")
    rs = resistance_features(i, traj, {int(x): float(x) for x in days}).rename(columns={"h": "day"})
    out = st.drop(columns=["h", "tau"]).merge(rs.drop(columns=["unc", "stale_days"], errors="ignore"), on=["ID", "day"], how="left")
    failed = bool(m.failed[i])
    out["failed"] = failed
    out["end_day"] = end_day
    for H in HS:
        if failed:
            out[f"y{H}"] = (end_day - out.day <= H).astype(int)
            out[f"valid{H}"] = True
        else:
            out[f"y{H}"] = 0
            out[f"valid{H}"] = out.day <= end_day - H
    return out


def main():
    m = load_meta()
    fold_of = {i: k for k, (_, te) in enumerate(folds(m)) for i in te}
    ids = sorted(zip_index())
    parts = Parallel(n_jobs=8)(delayed(one)(i, m, fold_of[i]) for i in ids)
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    df["activation"] = df.ID.map(m.activation_time.map(lambda x: x.timestamp()))
    df.to_parquet(ROOT / "data" / "landmarks.parquet")
    print(len(df), "landmarks;", df.ID.nunique(), "batteries;", df[["ID", "failed"]].drop_duplicates().failed.mean())
    return df


if __name__ == "__main__":
    main()
