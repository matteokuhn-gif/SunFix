"""Phase 3 stage C: EOL classifiers with battery-level 5-fold CV, balanced accuracy, calibration, false alarms."""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import HORIZONS, SETS
from .load import ROOT, load_meta
from .phase3_features import N_FOLDS, folds

MODELS = ("gpc", "lr", "hgb")


def make_model(name: str, d: int):
    if name == "gpc":   # SE kernel with automatic relevance determination, as in Aitio (sklearn GPC)
        return GaussianProcessClassifier(ConstantKernel(1.0) * RBF(length_scale=np.full(d, 3.0), length_scale_bounds=(1e-1, 1e3)),
                                         n_restarts_optimizer=0, random_state=0)
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)


def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, 0)
    Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    lo, hi = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)    # winsorise: single resistance spikes must not dominate
    Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def one(h, k, set_name, model, tr, te, tab, y):
    cols = SETS[set_name]
    a, b = tab.loc[tr, cols].to_numpy(float), tab.loc[te, cols].to_numpy(float)
    a, b = prep(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = make_model(model, len(cols)).fit(a, y[tr])
    return dict(h=h, fold=k, set=set_name, model=model, ID=te, p=mdl.predict_proba(b)[:, 1])


def metrics(y, p, thr=0.5, n_boot=1000, seed=0):
    pred = p >= thr
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y)); yy, pp = y[k], pred[k]
        bs.append(0.5 * ((pp & yy).sum() / max(yy.sum(), 1) + (~pp & ~yy).sum() / max((~yy).sum(), 1)))
    bins = np.quantile(p, np.linspace(0, 1, 11)); bins[-1] += 1e-9
    ece = sum(abs(p[(p >= lo) & (p < hi)].mean() - y[(p >= lo) & (p < hi)].mean()) * ((p >= lo) & (p < hi)).mean()
              for lo, hi in zip(bins[:-1], bins[1:]) if ((p >= lo) & (p < hi)).any())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sensitivity=sens, specificity=spec, balanced_acc=0.5 * (sens + spec),
                ba_lo=np.percentile(bs, 2.5), ba_hi=np.percentile(bs, 97.5), precision=tp / max(tp + fp, 1),
                false_alarms_per_1000_per_month=1000 * fp / max(fp + tn, 1), brier=float(np.mean((p - y) ** 2)), ece=float(ece))


def main(models=MODELS, sets=("res", "stress", "res+stress", "age_only", "res_core", "res_core+stress", "stale_only")):
    m = load_meta(); fl = folds(m); y_all = m.failed
    stress = pd.read_parquet(ROOT / "data" / "features_stress.parquet")
    jobs = []
    for k, (tr, te) in enumerate(fl):
        res = pd.read_parquet(ROOT / "data" / f"features_res_f{k}.parquet")
        for h in HORIZONS:
            tab = stress[stress.h == h].set_index("ID").join(res[res.h == h].set_index("ID").drop(columns="h"), how="outer")
            y = y_all.reindex(tab.index).to_numpy()
            tab = tab.reset_index().set_index("ID", drop=False); tab.index = tab.ID.to_numpy(); yy = pd.Series(y, index=tab.index)
            for s in sets:
                for mod in models:
                    if mod == "gpc" and s in ("age_only", "stale_only"):
                        continue
                    jobs.append((h, k, s, mod, tr, te, tab, yy))
    t = time.time()
    out = Parallel(n_jobs=8, verbose=0)(delayed(one)(h, k, s, mod, tr, te, tab, yy) for h, k, s, mod, tr, te, tab, yy in jobs)
    print("fit", len(jobs), "models", f"{time.time() - t:.0f}s", flush=True)
    pred = pd.concat([pd.DataFrame(dict(h=o["h"], fold=o["fold"], set=o["set"], model=o["model"], ID=o["ID"], p=o["p"])) for o in out])
    pred["failed"] = y_all.reindex(pred.ID).to_numpy()
    pred.to_parquet(ROOT / "results" / "phase3_predictions.parquet")
    rows = []
    for (h, s, mod), g in pred.groupby(["h", "set", "model"]):
        r = metrics(g.failed.to_numpy(bool), g.p.to_numpy()); r.update(h=h, set=s, model=mod)
        fold_ba = [metrics(f.failed.to_numpy(bool), f.p.to_numpy(), n_boot=1)["balanced_acc"] for _, f in g.groupby("fold")]
        r["ba_fold_sd"] = float(np.std(fold_ba)); rows.append(r)
    res = pd.DataFrame(rows)[["h", "set", "model", "balanced_acc", "ba_lo", "ba_hi", "ba_fold_sd", "sensitivity", "specificity", "precision",
                              "false_alarms_per_1000_per_month", "TP", "FP", "FN", "TN", "brier", "ece"]]
    res.to_csv(ROOT / "results" / "phase3_results.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    r = main()
    print(r[r.model.isin(["gpc", "lr"])].pivot_table(index=["set", "model"], columns="h", values="balanced_acc").round(3))
