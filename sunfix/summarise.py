"""One streaming pass over all batteries: per-battery summary statistics (no caching of raw data)."""
import numpy as np
import pandas as pd

from .load import ROOT, load_meta, read_raw, zip_index


def summarise_one(i: int) -> dict:
    d = read_raw(i)
    t = d["t"].to_numpy()
    dt = np.diff(t)
    ch = d["I"] < -0.2  # charging candidate under the 'negative = charging' reading
    dis = d["I"] > 0.2
    v = d["V"].to_numpy()
    dV = np.diff(v)
    dI = np.diff(d["I"].to_numpy())
    big = np.abs(dI) > 0.5  # current steps, to read the sign of dV/dI
    return dict(
        ID=i, rows=len(d), t0=t[0], t1=t[-1], unsorted=int((dt < 0).sum()), dup_t=int((dt == 0).sum()),
        nan_I=int(d["I"].isna().sum()), nan_V=int(d["V"].isna().sum()), nan_T=int(d["T"].isna().sum()),
        dt_med=float(np.median(dt)), dt_p99=float(np.percentile(dt, 99)), gap_max_d=float(dt.max() / 86400),
        gaps_gt1d=int((dt > 86400).sum()), gaps_gt30d=int((dt > 30 * 86400).sum()),
        I_min=d["I"].min(), I_max=d["I"].max(), V_min=v.min(), V_max=v.max(), T_min=d["T"].min(), T_max=d["T"].max(),
        I_p01=d["I"].quantile(0.01), I_p99=d["I"].quantile(0.99), V_mean=v.mean(), T_mean=d["T"].mean(),
        frac_I_neg=float((d["I"] < -0.05).mean()), frac_I_pos=float((d["I"] > 0.05).mean()),
        V_mean_when_Ineg=float(d["V"][ch].mean()) if ch.any() else np.nan,
        V_mean_when_Ipos=float(d["V"][dis].mean()) if dis.any() else np.nan,
        dV_dI_corr=float(np.corrcoef(dV[big], dI[big])[0, 1]) if big.sum() > 10 else np.nan,
        n_T_const_runs=int((np.diff(d["T"].to_numpy()) == 0).mean() * 100),
    )


def run(out=ROOT / "data" / "summary_all.csv") -> pd.DataFrame:
    rows = [summarise_one(i) for i in sorted(zip_index())]
    s = pd.DataFrame(rows).set_index("ID")
    s.to_csv(out)
    return s


if __name__ == "__main__":
    run()
