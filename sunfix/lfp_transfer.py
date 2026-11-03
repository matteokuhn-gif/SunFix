"""Phase 6d.4: lead-acid -> LFP transfer, rank agreement only (no LFP labels used for training or tuning).
A LightGBM trained on the Bboxx landmarks (training cohort, 30-day label) with ONLY baseline-relative resistance features (level / baseline and slopes over
7 / 30 / 90 d divided by baseline per day; age and usage features are dropped because they are not comparable between a 12 V lead-acid battery and a
24 V LFP pack) is applied to the pack-only LFP resistance trajectories (per-system normalisation = division by the system's own baseline).
Question asked: does the Bboxx model's risk rank LFP systems the way the per-cell method does? Metrics: Spearman correlation across systems of the mean
predicted risk over the last 30 days vs the per-cell maximum drift, and AUC of that risk for faults vs controls (pseudo-labels from 6d.2).
Run: python -m sunfix.lfp_transfer  (about 3 min). Output: results/lfp_transfer_rank_agreement.csv"""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr

from .eval_ops import split
from .lfp_pack import ROOT, Spec, trajectory
from .lfp_pack_run import SYSTEMS, auc, labels, prep_system

warnings.filterwarnings("ignore")
FEATS = ["rel_level", "rel_slope7", "rel_slope30", "rel_slope90"]


def bboxx_model():
    df = pd.read_parquet(ROOT / "data" / "landmarks6.parquet")
    base = df.level / df.rel_level
    for w in (7, 30, 90):
        df[f"rel_slope{w}"] = df[f"slope{w}"] / 1000.0 / base          # fraction of baseline per day
    sp = split(df)
    tr = df[df.ID.isin(sp["train"]) & df.valid30]
    m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.9,
                       random_state=0, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1).fit(tr[FEATS], tr.y30)
    te = df[df.ID.isin(sp["test"]) & df.valid30 & (df.age >= 360)]
    from sklearn.metrics import roc_auc_score
    return m, float(roc_auc_score(te.y30, m.predict_proba(te[FEATS])[:, 1])), {k: (float(df[k].quantile(0.01)), float(df[k].quantile(0.99))) for k in FEATS}


def lfp_features(sid, hyp):
    p = prep_system(sid, Spec())
    if p is None:
        return None
    tr = trajectory(p, hyp)
    day, r = tr.day.to_numpy(), tr.r0.to_numpy()
    b0 = max(30.0, float(day.min())); b = (day >= b0) & (day <= b0 + 30)
    if b.sum() < 3 or np.median(r[b]) <= 0:
        return None
    base = np.median(r[b]); rows = []
    for d in day[day > b0 + 30]:
        row = dict(day=d, rel_level=r[np.searchsorted(day, d, side="right") - 1] / base)
        for w in (7, 30, 90):
            sel = (day <= d) & (day >= d - w)
            row[f"rel_slope{w}"] = np.polyfit(day[sel], r[sel], 1)[0] / base if sel.sum() >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    m, auc_bboxx, rng = bboxx_model()
    print(f"Bboxx test AUC (at-risk, relative-resistance features only, no age): {auc_bboxx:.3f}", flush=True)
    hyp = json.load(open(ROOT / "data" / "battgp_pack" / "hyper_full.json"))
    lab = labels()
    feats = Parallel(n_jobs=6)(delayed(lfp_features)(s, hyp) for s in SYSTEMS)
    rows = []
    for s, f in zip(SYSTEMS, feats):
        if f is None or len(f) < 5:
            continue
        for k, (lo, hi) in rng.items():
            f[k] = f[k].clip(lo, hi)                      # keep LFP features inside the range the model saw
        risk = m.predict_proba(f[FEATS])[:, 1]
        last = f.day >= f.day.max() - 30
        rows.append(dict(system=s, role=lab.loc[s, "role"], max_cell_drift_mohm=lab.loc[s, "max_drift_mohm_acausal"], mean_risk_last30d=float(risk[last].mean()), max_risk=float(risk.max()),
                         end_rel_level=float(f.rel_level[last].mean()), n_days=len(f)))
    t = pd.DataFrame(rows); ev = t[t.role.isin(["fault", "control"])]
    rho, p = spearmanr(ev.mean_risk_last30d, ev.max_cell_drift_mohm)
    rng_ = np.random.default_rng(0); boots = []
    for _ in range(2000):
        i = rng_.integers(0, len(ev), len(ev)); b = ev.iloc[i]
        if b.max_cell_drift_mohm.nunique() > 1 and b.mean_risk_last30d.nunique() > 1:
            boots.append(spearmanr(b.mean_risk_last30d, b.max_cell_drift_mohm)[0])
    a = auc(ev[ev.role == "fault"].mean_risk_last30d.tolist(), ev[ev.role == "control"].mean_risk_last30d.tolist())
    summary = dict(n_systems=len(ev), n_fault=int((ev.role == "fault").sum()), n_control=int((ev.role == "control").sum()), spearman_risk_vs_cell_drift=float(rho), spearman_p=float(p),
                   spearman_ci_lo=float(np.percentile(boots, 2.5)), spearman_ci_hi=float(np.percentile(boots, 97.5)), auc_fault_vs_control=a, bboxx_test_auc_same_features=auc_bboxx)
    print(t.round(3).to_string(index=False)); print(summary)
    t.to_csv(ROOT / "results" / "lfp_transfer_by_system.csv", index=False)
    pd.DataFrame([summary]).to_csv(ROOT / "results" / "lfp_transfer_rank_agreement.csv", index=False)


if __name__ == "__main__":
    main()
