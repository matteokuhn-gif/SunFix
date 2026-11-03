"""Phase 6b: benchmark models on the same landmarks, same temporal split and same ops evaluation as 6a. HYPOTHETICAL costs.
Run: python -m sunfix.bench   (about 10-15 min)

Models: (i) GP classifier (sklearn, ARD-RBF, on 2,000 sampled training landmarks: exact GPC cannot take 200k rows);
(ii) LightGBM, one binary model per horizon on daily landmarks ("fails within 30 / 56 d", discrete-time hazard form);
(iii) Cox PH and random survival forest (scikit-survival) trained on landmarks every 14 days with right-censoring of alive batteries;
(iv) unsupervised CUSUM + fleet z-score (rank average, ranks from the training cohort, no labels); (v) 2-of-3 ensemble (median of the
calibrated scores of LightGBM, RSF and CUSUM+z, i.e. alert iff >= 2 of 3 are above threshold). Ablations: LightGBM usage-only,
LightGBM + availability features (leak check). Bayesian online change-point detection is NOT implemented (CUSUM only).
All scores are Platt-calibrated on the calibration cohort. Operating point (tau, N) chosen on the calibration cohort, reported on test.
"""
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from .classify import make_model, prep
from .eval_ops import FIG, RES, SEED, TAUS, Cohort, costs, derived, evaluate, split
from .features import RES_CORE
from .load import ROOT
from .run_6a import best_cfg

warnings.filterwarnings("ignore")
if not hasattr(np, "trapezoid"):   # scikit-survival 0.28 needs numpy>=2 API; env is pinned to numpy 1.26 for BattGP/torch
    np.trapezoid = np.trapz
CACHE = ROOT / "data" / "bench_cache.pkl"
RES_F = RES_CORE + ["slope7", "slope30", "slope90", "curv90", "cusum", "fleet_z"]
USE_F = ["age", "ah", "mean_T", "mean_V", "cycles", "float_h", "efc", "ah_band_30", "eod_v_30", "low_h_30", "lvd_30", "hot35_30", "hot40_30"]
AVAIL_F = ["cover_30", "outage_days_30", "gap_h_30", "res_var"]
HEAD_F = RES_F + USE_F
HZ = (30, 56)


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(raw, y, cal_mask):
    m = LogisticRegression(C=1e6).fit(lg(raw[cal_mask])[:, None], y[cal_mask])
    return m.predict_proba(lg(raw)[:, None])[:, 1]


def med_impute(X, ref):
    med = np.nanmedian(ref, 0)
    return np.where(np.isnan(X), med, X)


def surv_to_prob(S_at, t):  # helper unused placeholder for clarity
    return 1 - S_at


def main():
    raw = costs(); c0 = derived(raw); W = raw["useful_window_days"]
    df = pd.read_parquet(ROOT / "data" / "landmarks6.parquet").sort_values(["ID", "day"]).reset_index(drop=True)
    sp = split(df)
    is_tr, is_ca, is_te = (df.ID.isin(sp[k]).to_numpy() for k in ("train", "calib", "test"))
    rem = (df.end_day - df.day).to_numpy().clip(min=1.0); ev = df.failed.to_numpy(bool)
    scores = {}   # name -> {H: raw score for all rows}

    def lgbm(name, feats):
        scores[name] = {}
        for H in HZ:
            tr = is_tr & df[f"valid{H}"].to_numpy()
            m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
                               colsample_bytree=0.7, random_state=SEED, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1)
            m.fit(df.loc[tr, feats], df.loc[tr, f"y{H}"])
            scores[name][H] = m.predict_proba(df[feats])[:, 1]
            if name == "LightGBM" and H == 30:
                shap_model = m
        return shap_model if name == "LightGBM" else None
    shap_model = lgbm("LightGBM", HEAD_F)
    lgbm("LightGBM usage-only", USE_F)
    lgbm("LightGBM age only", ["age"])
    lgbm("LightGBM + availability (leak check)", HEAD_F + AVAIL_F)
    print("lightgbm done", flush=True)

    # GP classifier on a sample of training landmarks
    rng = np.random.default_rng(SEED)
    scores["GP classifier (2,000-landmark sample)"] = {}
    for H in HZ:
        idx = np.flatnonzero(is_tr & df[f"valid{H}"].to_numpy())
        y = df[f"y{H}"].to_numpy()
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        sel = np.r_[rng.choice(pos, min(len(pos), 600), replace=False), rng.choice(neg, 1400, replace=False)]
        Xa = df.loc[sel, HEAD_F].to_numpy(float)
        gpc_in = prep(Xa, Xa)[0]
        med = np.nanmedian(Xa, 0); lo, hi = np.percentile(np.where(np.isnan(Xa), med, Xa), 1, 0), np.percentile(np.where(np.isnan(Xa), med, Xa), 99, 0)
        mu, sd = np.clip(np.where(np.isnan(Xa), med, Xa), lo, hi).mean(0), np.clip(np.where(np.isnan(Xa), med, Xa), lo, hi).std(0) + 1e-9
        gp = make_model("gpc", len(HEAD_F)).fit(gpc_in, y[sel])
        out = np.full(len(df), np.nan)
        need = np.flatnonzero(is_ca | is_te)
        for a in range(0, len(need), 8000):
            ii = need[a:a + 8000]
            X = np.clip(np.where(np.isnan(df.loc[ii, HEAD_F].to_numpy(float)), med, df.loc[ii, HEAD_F].to_numpy(float)), lo, hi)
            out[ii] = gp.predict_proba((X - mu) / sd)[:, 1]
        scores["GP classifier (2,000-landmark sample)"][H] = out
    print("gpc done", flush=True)

    # survival models: landmarks every 14 days for training
    sub = is_tr & (df.day.to_numpy() % 14 == 0)
    Xs_all = df[HEAD_F].to_numpy(float)
    ref = Xs_all[sub]
    lo, hi = np.nanpercentile(ref, 1, 0), np.nanpercentile(ref, 99, 0)
    Xi = np.clip(med_impute(Xs_all, ref), lo, hi)
    mu, sd = Xi[sub].mean(0), Xi[sub].std(0) + 1e-9
    Z = (Xi - mu) / sd
    ysurv = Surv.from_arrays(event=ev, time=rem)
    cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Z[sub], ysurv[sub])
    lp = cox.predict(Z)
    H0 = cox.cum_baseline_hazard_
    scores["Cox PH (landmark)"] = {H: 1 - np.exp(-H0(float(H)) * np.exp(lp)) for H in HZ}
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=100, max_features="sqrt", n_jobs=8, random_state=SEED).fit(Z[sub], ysurv[sub])
    ut = rsf.unique_times_
    pr = {H: np.full(len(df), np.nan) for H in HZ}
    need = np.flatnonzero(is_ca | is_te)
    for a in range(0, len(need), 20000):
        ii = need[a:a + 20000]
        S = rsf.predict_survival_function(Z[ii], return_array=True)
        for H in HZ:
            pr[H][ii] = 1 - S[:, max(np.searchsorted(ut, H, side="right") - 1, 0)]
    scores["Random survival forest (landmark)"] = pr
    print("survival done", flush=True)
    ibs = {}
    t14_ = df.day.to_numpy() % 14 == 0
    ytr_ = ysurv[is_tr & t14_]
    for name, mdl in (("Cox PH (landmark)", "cox"), ("Random survival forest (landmark)", "rsf")):
        try:
            tt = np.flatnonzero(is_te & t14_); tt = tt[rem[tt] > 60][:4000]; times = np.arange(7, 57, 7.0)
            if mdl == "cox":
                S = np.exp(-np.outer(np.exp(lp[tt]), H0(times)))
            else:
                Sfull = rsf.predict_survival_function(Z[tt], return_array=True)
                S = Sfull[:, [max(np.searchsorted(ut, q, side="right") - 1, 0) for q in times]]
            ibs[name] = float(integrated_brier_score(ytr_, ysurv[tt], S, times))
        except Exception as e:
            print("ibs failed", name, repr(e)); ibs[name] = np.nan

    # unsupervised CUSUM + fleet z (ranks fitted on training cohort, no labels)
    def rank01(x, ref):
        r = np.sort(ref[~np.isnan(ref)])
        return np.where(np.isnan(x), 0.5, np.searchsorted(r, x) / len(r))
    u = 0.5 * (rank01(df.cusum.to_numpy(float), df.cusum.to_numpy(float)[is_tr]) + rank01(df.fleet_z.to_numpy(float), df.fleet_z.to_numpy(float)[is_tr]))
    scores["CUSUM + fleet z-score (unsupervised)"] = {H: u for H in HZ}

    # calibration per model/horizon, ensemble, metrics
    cal = {}
    for name, d in scores.items():
        cal[name] = {H: platt(np.nan_to_num(d[H], nan=0.0), df[f"y{H}"].to_numpy(), is_ca & df[f"valid{H}"].to_numpy()) for H in HZ}
    ens = "2-of-3 ensemble (LightGBM, RSF, CUSUM+z)"
    parts = [cal["LightGBM"], cal["Random survival forest (landmark)"], cal["CUSUM + fleet z-score (unsupervised)"]]
    cal[ens] = {H: np.median(np.vstack([p[H] for p in parts]), 0) for H in HZ}
    scores[ens] = cal[ens]
    # survival-metric subsamples
    t14 = df.day.to_numpy() % 14 == 0
    ytr, yte = ysurv[is_tr & t14], ysurv[is_te & t14]
    rows = []
    for name in cal:
        for H in HZ:
            s, y, val = cal[name][H], df[f"y{H}"].to_numpy().astype(bool), df[f"valid{H}"].to_numpy()
            c, t = is_ca & val, is_te & val
            ba = lambda thr, m_: 0.5 * (((s[m_] >= thr) & y[m_]).sum() / y[m_].sum() + ((s[m_] < thr) & ~y[m_]).sum() / (~y[m_]).sum())
            thr = max(TAUS, key=lambda q: ba(q, c))
            tp, fp, fn, tn = (((s[t] >= thr) & y[t]).sum(), ((s[t] >= thr) & ~y[t]).sum(), ((s[t] < thr) & y[t]).sum(), ((s[t] < thr) & ~y[t]).sum())
            r = dict(model=name, horizon_days=H, threshold_ba_opt_on_calib=float(thr), balanced_accuracy=0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                     sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), precision_at_test_prevalence=tp / max(tp + fp, 1),
                     brier=float(np.mean((s[t] - y[t]) ** 2)), base_rate=float(y[t].mean()))
            tt = is_te & t14
            risk = cal[name][56][tt]
            try:
                r["c_index"] = float(concordance_index_censored(ev[tt], rem[tt], risk)[0])
            except Exception as e:
                print("c-index failed", name, repr(e)); r["c_index"] = np.nan
            try:
                mask = rem[tt] < ytr["time"].max()
                r["td_auc_at_horizon"] = float(cumulative_dynamic_auc(ytr, ysurv[tt][mask], risk[mask], [H])[0][0])
            except Exception as e:
                print("td-auc failed", name, repr(e)); r["td_auc_at_horizon"] = np.nan
            # at-risk subset (age >= 360 d: no battery ever failed younger), threshold re-chosen on the calibration cohort
            ar = df.age.to_numpy() >= 360
            ca_, te_ = c & ar, t & ar
            thr2 = max(TAUS, key=lambda q: ba(q, ca_))
            sens2 = ((s[te_] >= thr2) & y[te_]).sum() / y[te_].sum(); spec2 = ((s[te_] < thr2) & ~y[te_]).sum() / (~y[te_]).sum()
            from sklearn.metrics import roc_auc_score
            r.update(ba_at_risk_age360=0.5 * (sens2 + spec2), sens_at_risk=sens2, spec_at_risk=spec2, auc_at_risk_age360=float(roc_auc_score(y[te_], s[te_])),
                     base_rate_at_risk=float(y[te_].mean()))
            rows.append(r)
    for r in rows:
        r["integrated_brier_7_56d"] = ibs.get(r["model"], np.nan)
    tab = pd.DataFrame(rows)
    # operational metrics at each model's cost-optimal point (H = 30 scores), K unlimited, confirmation off and on
    ops = []
    base = df[["ID", "day", "failed", "end_day", "activation"]].copy()
    for name in cal:
        base["cal"] = cal[name][30]
        ca, te = Cohort(base, sp["calib"], "cal", W), Cohort(base, sp["test"], "cal", W)
        for conf in (False, True):
            import itertools
            best = min(({"tau": t_, "N": n_, **evaluate(ca, t_, n_, conf, None, c0)} for t_, n_ in itertools.product(TAUS, (1, 3, 5, 7))), key=lambda r_: r_["EC_per_1000_month"])
            r = evaluate(te, best["tau"], best["N"], conf, None, c0)
            ops.append(dict(model=name, horizon_days=30, ops_confirm="on" if conf else "off", ops_tau=best["tau"], ops_N=best["N"],
                            **{f"ops_{k}": v for k, v in r.items() if k in ("EC_per_1000_month", "false_dispatch_per_1000_month", "missed_per_1000_month", "true_dispatch_per_1000_month",
                                                                           "precision", "recall", "lead_median", "lead_p10", "lead_share_ge30", "flips_per_system_month")}))
    ops = pd.DataFrame(ops)
    out = tab.merge(ops, on=["model", "horizon_days"], how="left")
    out["cost_status"] = np.where(out.ops_EC_per_1000_month.notna(), "ops columns use HYPOTHETICAL costs", "")
    out.to_csv(RES / "benchmark_table.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tab.round(3).to_string(index=False))
    print(ops.round(2).to_string(index=False))
    # SHAP via LightGBM native contributions (TreeSHAP), on a test sample
    te_idx = np.flatnonzero(is_te & df.valid30.to_numpy())
    samp = rng.choice(te_idx, 20000, replace=False)
    contrib = shap_model.predict(df.loc[samp, HEAD_F], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=HEAD_F).sort_values()
    imp.sort_values(ascending=False).rename("mean_abs_shap_log_odds").to_csv(RES / "shap_lightgbm_bboxx.csv")
    fig, a = plt.subplots(figsize=(6.5, 5.5)); imp.plot.barh(ax=a, color="#1f5fa8"); a.set_xlabel("mean |SHAP| (log-odds), 30-day LightGBM, test sample"); fig.tight_layout()
    fig.savefig(FIG / "shap_lightgbm_bboxx.png", dpi=150)
    print(imp.sort_values(ascending=False).round(3).head(10).to_dict())


if __name__ == "__main__":
    main()
