"""Phase 3 stage A/B: per-fold resistance model (hyperparameters fitted on training-fold batteries only) and features."""
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .features import HORIZONS, end_time, resistance_features, stress_features
from .clean import clean_raw
from .fit_hyper import fit
from .load import ROOT, load_meta, zip_index
from .resistance import Normaliser, prepare, run_battery

N_FOLDS, SEED = 5, 0


def folds(m: pd.DataFrame):
    ids = np.array(sorted(m.index))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return [(ids[tr], ids[te]) for tr, te in skf.split(ids, m.failed.loc[ids].to_numpy())]


def taus_days(i, m):
    d = clean_raw(i, m)
    t_end = end_time(i, m, d); act = m.activation_time[i].timestamp()
    return {h: (t_end - h * 86400.0 - act) / 86400.0 for h in HORIZONS}


def res_one(i, m, h, norm, k):
    p = prepare(i, "ocv", m)
    if p is None:
        return pd.DataFrame([dict(ID=i, h=hh) for hh in HORIZONS])
    traj, _, _ = run_battery(p, norm, h["sf0"], h["sf1"], tuple(h["ls"]), tau=h["tau"])
    d = ROOT / "data" / f"resistance_f{k}"; d.mkdir(exist_ok=True)
    traj.to_parquet(d / f"{i}.parquet", index=False)
    return resistance_features(i, traj, taus_days(i, m))


def main():
    m = load_meta()
    ids = sorted(zip_index())
    t0 = time.time()
    stress = pd.concat(Parallel(n_jobs=8)(delayed(stress_features)(i, m) for i in ids), ignore_index=True)
    stress.to_parquet(ROOT / "data" / "features_stress.parquet"); print("stress features", time.time() - t0, flush=True)
    for k, (tr, te) in enumerate(folds(m)):
        t = time.time()
        h = fit("ocv", pool=tr.tolist(), out_path=ROOT / "data" / f"hyper_f{k}.json", quiet=True, seed=SEED + k)
        norm = Normaliser(np.array(h["norm_mean"]), np.array(h["norm_std"]), np.array(h["basis"]))
        res = pd.concat(Parallel(n_jobs=8)(delayed(res_one)(i, m, h, norm, k) for i in ids), ignore_index=True)
        res.to_parquet(ROOT / "data" / f"features_res_f{k}.parquet")
        print(f"fold {k}: hyper {({x: round(h[x], 2) for x in ('sf0', 'sf1', 'tau')})} ls {np.round(h['ls'], 1).tolist()} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
