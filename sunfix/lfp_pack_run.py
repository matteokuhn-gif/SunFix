"""Phase 6d.2: pack-only, OCV-free resistance GP on all 21 modelled LFP systems vs per-cell pseudo-labels.
Run: python -m sunfix.lfp_pack_run  (about 3 min). Outputs: results/lfp_pack_only_by_system.csv, results/lfp_pack_only_detection.csv.
Pseudo-labels: `results/lfp_percell_pseudolabels.csv` (per-cell drift >= 0.33 mOhm vs own days 30-60 baseline, smoothed trajectories; system 1 excluded as a
data artefact; faults = onset found, controls = no onset and max drift <= 0.5 mOhm). Hyperparameters are fitted label-free on 4 randomly chosen systems.
"""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .lfp_pack import Spec, alarm_days, rise_days, degrade, fit_hypers, load_pack, prepare_pack, section_start, trajectory, ROOT

warnings.filterwarnings("ignore")
HS = (0.05, 0.10, 0.20, 0.30, 0.50)   # relative pack-resistance rise over baseline that triggers an alarm
SYSTEMS = list(range(1, 22))
EARLY = 30.0          # an alarm more than 30 days before the per-cell onset counts as a false flag


def labels() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "results" / "lfp_percell_pseudolabels.csv").set_index("system")
    t["role"] = np.where(~t.eligible | (t.index == 1), "excluded", np.where(t.onset_acausal.notna(), "fault", np.where(t.max_drift_mohm_acausal <= 0.5, "control", "excluded")))
    return t


def prep_system(sid, spec):
    d = load_pack(sid)
    if len(d) < 100:
        return None
    return prepare_pack(degrade(d, spec, sid), spec, section_start(sid))


def one(sid, spec, hyp, hs=HS, history=None):
    p = prep_system(sid, spec)
    if p is None:
        return dict(system=sid, n_obs=0, alarms={h: np.nan for h in hs}, baseline=False, end_over_baseline=np.nan)
    tr = trajectory(p, hyp)
    a = rise_days(tr, list(hs), history=spec.history)
    day = tr.day.to_numpy()
    b0 = max(30.0, float(day.min()))
    r = tr.r0.to_numpy(); b = (day >= b0) & (day <= b0 + 30)
    ratio = float(np.median(r[day >= day.max() - 14.0]) / np.median(r[b])) if b.sum() >= 3 and np.median(r[b]) > 0 else np.nan   # end / baseline, threshold-free
    return dict(system=sid, n_obs=len(p.y), alarms=a, baseline=bool(b.sum() >= 3), end_day=float(day.max()), end_over_baseline=ratio)


def fit_hyp(spec, seed=0, pool=None):
    rng = np.random.default_rng(seed)
    lab = labels()
    pool = pool or [s for s in SYSTEMS if lab.loc[s, "eligible"]]
    chosen = sorted(rng.choice(pool, 4, replace=False).tolist())
    preps = [p for p in (prep_system(s, spec) for s in chosen) if p is not None]
    h = fit_hypers(preps, preps[0].X.shape[1])
    h["fit_systems"] = chosen
    return h


def evaluate(results, hs=HS):
    lab = labels()
    rows = []
    for h in hs:
        det, delay, early, ctrl_fa, nobase, nF, nC = 0, [], 0, 0, 0, 0, 0
        for r in results:
            role = lab.loc[r["system"], "role"]
            a = r["alarms"][h]
            if role == "fault":
                nF += 1
                if not r["baseline"]:
                    nobase += 1
                on = lab.loc[r["system"], "onset_acausal"]
                if a == a:
                    if a < on - EARLY:
                        early += 1
                    else:
                        det += 1; delay.append(a - on)
            elif role == "control":
                nC += 1
                ctrl_fa += int(a == a)
        d = np.array(delay)
        rows.append(dict(rise_threshold=h, n_fault_systems=nF, n_control_systems=nC, detected=det, detection_rate=det / max(nF, 1), early_flags_on_fault_systems=early,
                         fault_systems_without_baseline=nobase, control_false_flags=ctrl_fa, control_false_flag_rate=ctrl_fa / max(nC, 1),
                         delay_median_days=float(np.median(d)) if len(d) else np.nan, delay_p10_days=float(np.percentile(d, 10)) if len(d) else np.nan,
                         delay_p90_days=float(np.percentile(d, 90)) if len(d) else np.nan))
    return pd.DataFrame(rows)


def auc(pos, neg):
    pos, neg = [x for x in pos if x is not None and x == x], [x for x in neg if x is not None and x == x]
    if not pos or not neg:
        return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def robustness(spec=None, seeds=(0, 1, 2), hs=(0.2, 0.3)):
    """Refit the label-free hyperparameters on different random sets of 4 systems; report detection, control false flags and the threshold-free AUC."""
    spec = spec or Spec(); lab = labels(); rows = []
    for sd in seeds:
        hyp = fit_hyp(spec, seed=sd)
        res = Parallel(n_jobs=6)(delayed(one)(s_, spec, hyp) for s_ in SYSTEMS)
        ev = evaluate(res, hs=hs)
        a = auc([r["end_over_baseline"] for r in res if lab.loc[r["system"], "role"] == "fault"], [r["end_over_baseline"] for r in res if lab.loc[r["system"], "role"] == "control"])
        for _, e in ev.iterrows():
            rows.append(dict(seed=sd, fit_systems=str(hyp["fit_systems"]), sf1=hyp["sf1"], tau=hyp["tau"], auc_end_over_baseline=a, **e.to_dict()))
    return pd.DataFrame(rows)


def main():
    spec = Spec()
    hyp = fit_hyp(spec)
    (ROOT / "data" / "battgp_pack").mkdir(exist_ok=True)
    json.dump(hyp, open(ROOT / "data" / "battgp_pack" / "hyper_full.json", "w"))
    print("hyperparameters", {k: (np.round(hyp[k], 3).tolist() if k in ("sf0", "sf1", "tau", "ls") else None) for k in ("sf0", "sf1", "tau", "ls")}, "fitted on systems", hyp["fit_systems"], flush=True)
    res = Parallel(n_jobs=6)(delayed(one)(s, spec, hyp) for s in SYSTEMS)
    lab = labels()
    by = pd.DataFrame([dict(system=r["system"], role=lab.loc[r["system"], "role"], per_cell_onset=lab.loc[r["system"], "onset_acausal"], per_cell_cell=lab.loc[r["system"], "cell_acausal"],
                            n_obs=r["n_obs"], has_baseline=r["baseline"], **{f"pack_alarm_h{h}": r["alarms"][h] for h in HS}) for r in res])
    by.to_csv(ROOT / "results" / "lfp_pack_only_by_system.csv", index=False)
    ev = evaluate(res)
    ev.to_csv(ROOT / "results" / "lfp_pack_only_detection.csv", index=False)
    rb = robustness()
    rb.to_csv(ROOT / "results" / "lfp_pack_only_robustness.csv", index=False)
    print(rb[["seed", "fit_systems", "sf1", "tau", "auc_end_over_baseline", "rise_threshold", "detected", "early_flags_on_fault_systems", "control_false_flags", "delay_median_days"]].round(2).to_string(index=False))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(by.round(1).to_string(index=False)); print(ev.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
