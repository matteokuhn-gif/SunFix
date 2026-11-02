"""Population-level hyperparameter fit for the resistance GP: maximise summed innovation log-likelihood
plus Aitio's inverse-gamma(a=1, b=2) prior on lengthscales. Label-free (no failure information is used).
The sigma_f half-normal prior of Aitio Eq. 7 is NOT used: with it their reported medians (sigma_f0 = 2.26) would be
implausible, so their sigma units must differ from ours; we bound sigma_f instead."""
import json
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .load import ROOT, dev_subset, load_meta
from .resistance import START, VARIANTS, fit_normaliser, prepare, run_battery

BOUNDS = dict(sf0=(0.01, 30.0), sf1=(0.01, 30.0), tau=(1e-3, 3.0), ls=(0.3, 60.0))


def unpack(theta, d):
    e = np.exp(theta)
    return float(e[0]), float(e[1]), float(e[2]), tuple(e[3:3 + d])


def neg_log_post(theta, preps, norm):
    d = norm.basis.shape[1]
    sf0, sf1, tau, ls = unpack(theta, d)
    if not (BOUNDS["sf0"][0] <= sf0 <= BOUNDS["sf0"][1] and BOUNDS["sf1"][0] <= sf1 <= BOUNDS["sf1"][1]
            and BOUNDS["tau"][0] <= tau <= BOUNDS["tau"][1]
            and all(BOUNDS["ls"][0] <= l <= BOUNDS["ls"][1] for l in ls)):
        return 1e12
    lls = Parallel(n_jobs=-1)(delayed(lambda p: run_battery(p, norm, sf0, sf1, ls, tau=tau)[1])(p) for p in preps)
    prior = sum(np.log(2.0) - 2 * np.log(l) - 2.0 / l for l in ls)   # inverse gamma(1, 2) log pdf
    return -(sum(lls) + prior)


def fit(variant: str, n_each: int = 4, stride: int = 10, maxiter: int = 600, seed: int = 0):
    m = load_meta(); ids = dev_subset()
    rng = np.random.default_rng(seed)
    failed = [i for i in ids if m.failed[i]]; alive = [i for i in ids if not m.failed[i]]
    train = sorted(rng.choice(failed, n_each, replace=False).tolist() + rng.choice(alive, n_each, replace=False).tolist())
    preps = [prepare(i, variant, m, stride=stride) for i in train]
    norm = fit_normaliser(preps)
    d = norm.basis.shape[1]
    ls0 = [START["ls"][0], START["ls"][1], START["ls"][2]] if d == 3 else [START["ls"][0], START["ls"][2]]
    theta0 = np.log([START["sf0"], START["sf1"], 0.05, *ls0])
    t = time.time()
    res = minimize(neg_log_post, theta0, args=(preps, norm), method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-1))
    sf0, sf1, tau, ls = unpack(res.x, d)
    out = dict(variant=variant, train_ids=train, sf0=sf0, sf1=sf1, tau=tau, ls=list(ls), nlp=float(res.fun),
               nlp_start=float(neg_log_post(theta0, preps, norm)), evals=int(res.nfev), seconds=time.time() - t,
               norm_mean=norm.mean.tolist(), norm_std=norm.std.tolist(), basis=norm.basis.tolist())
    (ROOT / "data" / f"hyper_{variant}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("basis",)}, indent=1))


if __name__ == "__main__":
    import sys
    fit(sys.argv[1])
