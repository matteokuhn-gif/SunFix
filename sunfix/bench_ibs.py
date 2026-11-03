"""Phase 6b: integrated Brier score (7-56 d) for the two survival models. Same data prep, seeds and models as sunfix.bench.
Separate script because the first bench run filtered out all test events (bug); refitting only Cox and RSF takes a few minutes."""
import warnings

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from .bench import HEAD_F, SEED, med_impute
from .eval_ops import RES, split
from .load import ROOT

warnings.filterwarnings("ignore")
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


def main():
    df = pd.read_parquet(ROOT / "data" / "landmarks6.parquet").sort_values(["ID", "day"]).reset_index(drop=True)
    sp = split(df)
    is_tr, is_te = (df.ID.isin(sp[k]).to_numpy() for k in ("train", "test"))
    rem = (df.end_day - df.day).to_numpy().clip(min=1.0); ev = df.failed.to_numpy(bool)
    sub = is_tr & (df.day.to_numpy() % 14 == 0)
    X = df[HEAD_F].to_numpy(float); ref = X[sub]
    lo, hi = np.nanpercentile(ref, 1, 0), np.nanpercentile(ref, 99, 0)
    Xi = np.clip(med_impute(X, ref), lo, hi); Z = (Xi - Xi[sub].mean(0)) / (Xi[sub].std(0) + 1e-9)
    y = Surv.from_arrays(event=ev, time=rem)
    cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Z[sub], y[sub]); H0 = cox.cum_baseline_hazard_
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=100, max_features="sqrt", n_jobs=8, random_state=SEED).fit(Z[sub], y[sub])
    tt = np.flatnonzero(is_te & (df.day.to_numpy() % 14 == 0))
    rng = np.random.default_rng(SEED); tt = np.sort(rng.choice(tt, min(6000, len(tt)), replace=False))
    times = np.arange(7, 57, 7.0)
    S_cox = np.exp(-np.outer(np.exp(cox.predict(Z[tt])), H0(times)))
    ut = rsf.unique_times_
    S_rsf = rsf.predict_survival_function(Z[tt], return_array=True)[:, [max(np.searchsorted(ut, q, side="right") - 1, 0) for q in times]]
    out = {"Cox PH (landmark)": float(integrated_brier_score(y[sub], y[tt], S_cox, times)),
           "Random survival forest (landmark)": float(integrated_brier_score(y[sub], y[tt], S_rsf, times))}
    # reference: Kaplan-Meier-free constant predictor = overall training survival is not provided; report the two models only
    pd.Series(out, name="integrated_brier_7_56d").to_csv(RES / "benchmark_ibs.csv")
    t = pd.read_csv(RES / "benchmark_table.csv")
    t["integrated_brier_7_56d"] = t.model.map(out)
    t.to_csv(RES / "benchmark_table.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
