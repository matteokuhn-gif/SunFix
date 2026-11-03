"""Phase 6c: split-conformal alert gate (static, age-conditional, adaptive, sliding-window) and the real effect of persistence N.
Run: python -m sunfix.conformal  (about 3 min). No cost values are used here: results do not depend on config/costs.yaml.

Score: 30-day LightGBM probability (as in 6b, trained on the training cohort only). Nonconformity of a system-day = its score; for persistence N the
statistic is the minimum score over the last N days (an alert fires iff all N days exceed the threshold).
'Healthy' system-day = no failure within the next 56 days and outcome observed (valid56 & y56 == 0), age >= 360 d (the at-risk period).
Calibration: healthy days of the calibration cohort. Evaluation: later (test-cohort) months. Labels arrive 56 days late, so every online
variant only uses healthy-day outcomes from dates <= t - 56.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from .bench import HEAD_F
from .eval_ops import FIG, RES, SEED, split
from .load import ROOT

ALPHAS = (0.01, 0.02, 0.05, 0.10)
NS = (1, 3, 5, 7)
DELAY = 56
CITED = {1: 25.3, 3: 6.4, 5: 1.0}   # Adnan et al. 2011 as quoted in PHASE6_PROMPT.md (percent, equal FAR/MAR); definition not verified here


def qhat(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))/n empirical quantile of healthy calibration scores."""
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def load():
    df = pd.read_parquet(ROOT / "data" / "landmarks6.parquet").sort_values(["ID", "day"]).reset_index(drop=True)
    sp = split(df)
    is_tr = df.ID.isin(sp["train"]).to_numpy()
    tr = is_tr & df.valid30.to_numpy()
    m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
                       colsample_bytree=0.7, random_state=SEED, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1)
    m.fit(df.loc[tr, HEAD_F], df.loc[tr, "y30"])
    df["s1"] = m.predict_proba(df[HEAD_F])[:, 1]
    g = df.groupby("ID").s1
    for N in NS[1:]:
        df[f"s{N}"] = g.transform(lambda x: x.rolling(N, min_periods=N).min())     # persistence statistic (days are contiguous)
    df["date"] = pd.to_datetime(df.activation + df.day * 86400, unit="s").dt.floor("D")
    df["month"] = df.date.dt.to_period("M").astype(str)
    df["healthy"] = df.valid56.to_numpy() & (df.y56.to_numpy() == 0)
    df["cohort"] = np.where(df.ID.isin(sp["calib"]), "calib", np.where(df.ID.isin(sp["test"]), "test", "train"))
    df["at_risk"] = df.age >= 360
    return df


def drift_tables(df):
    cal = df[(df.cohort == "calib") & df.healthy & df.at_risk]
    te = df[(df.cohort == "test") & df.healthy & df.at_risk]
    s_cal = cal.s1.to_numpy()
    rows, series = [], {}
    dates = pd.date_range(te.date.min(), te.date.max())
    for a in ALPHAS:
        q0 = qhat(s_cal, a)
        thr = {"static": pd.Series(q0, index=dates)}
        # age-conditional (Mondrian): threshold per 60-day age bin from calibration healthy days
        qb = {b: qhat(g.s1.to_numpy(), a) for b, g in cal.groupby((cal.age // 60) * 60) if len(g) >= 200}
        te_q = ((te.age // 60) * 60).map(qb).fillna(q0).to_numpy()
        # adaptive conformal inference with delayed labels
        gamma, al, a_t, err_hist = 0.02, a, {}, {}
        thr_aci = {}
        for d in dates:
            thr_aci[d] = qhat(s_cal, al)
            dd = d - pd.Timedelta(days=DELAY)
            if dd in err_hist:
                al = float(np.clip(al + gamma * (a - err_hist[dd]), 0.002, 0.5))
            day = te[te.date == d]
            err_hist[d] = float((day.s1 > thr_aci[d]).mean()) if len(day) else a
        thr["adaptive (ACI, delayed labels)"] = pd.Series(thr_aci)
        # sliding window: healthy outcomes known by t-56, from the previous 90 days, calibration + test-cohort batteries seen so far
        pool = df[df.healthy & df.at_risk & (df.cohort != "train")]
        sw = {}
        for d in dates:
            w = pool[(pool.date <= d - pd.Timedelta(days=DELAY)) & (pool.date > d - pd.Timedelta(days=DELAY + 90))]
            sw[d] = qhat(w.s1.to_numpy(), a) if len(w) >= 300 else q0
        thr["sliding window (90 d, delayed)"] = pd.Series(sw)
        for name, t in thr.items():
            tv = te.date.map(t).to_numpy(float)
            alert = te.s1.to_numpy() > tv
            for mo, mm in te.assign(alert=alert).groupby("month"):
                rows.append(dict(alpha=a, method=name, month=mo, healthy_days=len(mm), realised_FAR=mm.alert.mean()))
            rows.append(dict(alpha=a, method=name, month="ALL", healthy_days=len(te), realised_FAR=alert.mean()))
        alert = te.s1.to_numpy() > te_q
        for mo, mm in te.assign(alert=alert).groupby("month"):
            rows.append(dict(alpha=a, method="age-conditional (Mondrian)", month=mo, healthy_days=len(mm), realised_FAR=mm.alert.mean()))
        rows.append(dict(alpha=a, method="age-conditional (Mondrian)", month="ALL", healthy_days=len(te), realised_FAR=alert.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "conformal_far_by_month.csv", index=False)
    return out


def persistence_table(df):
    cal = df[(df.cohort == "calib") & df.healthy & df.at_risk]
    te = df[(df.cohort == "test") & df.at_risk]
    teh = te[te.healthy]
    # equal-error threshold for N=1 on the calibration cohort (healthy vs failure-within-56-days days)
    cal_pos = df[(df.cohort == "calib") & df.at_risk & df.valid56 & (df.y56 == 1)]
    grid = np.quantile(cal.s1, np.linspace(0.5, 0.999, 400))
    eer = min(grid, key=lambda q: abs((cal.s1 > q).mean() - (cal_pos.s1 <= q).mean()))
    rows = []
    # (a) at a fixed N=1 threshold (equal error), what does persistence do to the healthy false-alarm rate?
    p1 = float((teh.s1 > eer).mean())
    for N in NS:
        emp = float((teh[f"s{N}"] > eer).mean())
        rows.append(dict(part="equal-error N=1 threshold", N=N, empirical_FPR=emp, iid_prediction=p1**N, ratio_empirical_to_iid=emp / max(p1**N, 1e-12),
                         reduction_factor_vs_N1=emp / p1, cited_Adnan_FPR_pct=CITED.get(N, np.nan), cited_reduction_vs_N1=(CITED[N] / CITED[1]) if N in CITED else np.nan,
                         threshold=float(eer)))
    # (b) at equal healthy-day false-alarm target (conformal per N), what do detection and lead time look like?
    for a in (0.02, 0.05, 0.10):
        for N in NS:
            q = qhat(cal[f"s{N}"].dropna().to_numpy(), a)
            far = float((teh[f"s{N}"] > q).mean())
            bat_fa = teh.assign(al=teh[f"s{N}"] > q).groupby("ID").al.max().mean()      # healthy at-risk batteries ever alerted
            inwin = te[te.valid56 & (te.y56 == 1)]
            det, leads = [], []
            for i, g in inwin.groupby("ID"):
                hit = g[g[f"s{N}"] > q]
                det.append(len(hit) > 0)
                if len(hit):
                    leads.append(float(hit.end_day.iloc[0] - hit.day.iloc[0]))
            leads = np.array(leads)
            rows.append(dict(part=f"conformal alpha={a}", N=N, empirical_FPR=far, target_alpha=a, healthy_batteries_ever_alerted=float(bat_fa),
                             detection_rate_failed_batteries=float(np.mean(det)), n_failed_test_batteries=len(det),
                             lead_median_days=float(np.median(leads)) if len(leads) else np.nan, lead_p10_days=float(np.percentile(leads, 10)) if len(leads) else np.nan,
                             lead_share_ge30=float((leads >= 30).mean()) if len(leads) else np.nan, threshold=float(q)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "persistence_bboxx.csv", index=False)
    return out


def figures(far, per):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a_, al in zip(ax, (0.05, 0.10)):
        d = far[(far.alpha == al) & (far.month != "ALL") & (far.healthy_days >= 500)]
        for meth, g in d.groupby("method"):
            a_.plot(g.month, g.realised_FAR, "o-", label=meth)
        a_.axhline(al, color="k", ls="--", lw=1, label=f"target {al:.0%}"); a_.set_title(f"alpha = {al}"); a_.tick_params(axis="x", rotation=45); a_.grid(alpha=.3)
    ax[0].set_ylabel("realised false-alarm rate on healthy system-days"); ax[0].legend(fontsize=7)
    fig.suptitle("Conformal gate: realised false-alarm rate by month (test cohort, age >= 360 d)"); fig.tight_layout(); fig.savefig(FIG / "conformal_far_by_month.png", dpi=150); plt.close(fig)
    p = per[per.part == "equal-error N=1 threshold"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(p.N, p.empirical_FPR * 100, "o-", label="empirical (autocorrelated)"); ax[0].plot(p.N, p.iid_prediction * 100, "s--", label="i.i.d. theory p^N")
    c = p.dropna(subset=["cited_Adnan_FPR_pct"]); ax[0].plot(c.N, c.cited_Adnan_FPR_pct, "^:", label="Adnan et al. 2011 (as cited)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("persistence N (consecutive days)"); ax[0].set_ylabel("false-positive rate, % of healthy system-days"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    q = per[per.part == "conformal alpha=0.05"]
    ax[1].plot(q.N, q.detection_rate_failed_batteries * 100, "o-", label="failed batteries alerted"); ax[1].plot(q.N, q.lead_median_days, "s-", label="median lead time (days)")
    ax[1].set_xlabel("persistence N"); ax[1].set_title("same 5% false-alarm budget at every N"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "persistence_effect.png", dpi=150); plt.close(fig)


def main():
    df = load()
    far = drift_tables(df)
    per = persistence_table(df)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(far[far.month == "ALL"].pivot_table(index="alpha", columns="method", values="realised_FAR").round(3).to_string())
    print(far[(far.alpha == 0.05) & (far.month != "ALL")].pivot_table(index="month", columns="method", values="realised_FAR").round(3).to_string())
    print(per.round(3).to_string(index=False))
    figures(far, per)


if __name__ == "__main__":
    main()
