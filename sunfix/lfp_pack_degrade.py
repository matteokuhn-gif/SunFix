"""Phase 6d.3: Phase 4-style degradations applied to the pack-only LFP model. Run: python -m sunfix.lfp_pack_degrade (about 12 min).
Hyperparameters: the seed-0 full-data fit, reused for every scenario (no_temperature refits because it has one input fewer).
Because the full-data pack-only baseline is unstable (see lfp_pack_only_robustness.csv), the detection columns are NOT reliable;
the coverage columns (systems with a baseline, observations kept) are solid. Output: results/minimum_data_table_lfp.csv."""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .lfp_pack import Spec, ROOT
from .lfp_pack_run import SYSTEMS, auc, evaluate, fit_hyp, labels, one

warnings.filterwarnings("ignore")
DELTA = 0.2
SCEN = {
    "native (about 5 s)": Spec("native"),
    "interval 1 min": Spec("i60", interval=60.0), "interval 5 min": Spec("i300", interval=300.0), "interval 15 min": Spec("i900", interval=900.0), "interval 1 h": Spec("i3600", interval=3600.0),
    "no temperature": Spec("noT", temp=False),
    "history 30 d": Spec("h30", history=30), "history 90 d": Spec("h90", history=90), "history 180 d": Spec("h180", history=180),
    "voltage rounded to 0.1 V": Spec("v01", v_round=0.1), "voltage rounded to 1 V": Spec("v1", v_round=1.0), "current rounded to 1 A": Spec("i1", i_round=1.0),
    "packet loss 10%": Spec("l10", loss=0.10), "packet loss 30%": Spec("l30", loss=0.30), "packet loss 50%": Spec("l50", loss=0.50),
}


def main():
    lab = labels(); rows = []
    hyp_full = json.load(open(ROOT / "data" / "battgp_pack" / "hyper_full.json"))
    for name, spec in SCEN.items():
        hyp = fit_hyp(spec, seed=0) if not spec.temp else hyp_full
        res = Parallel(n_jobs=6)(delayed(one)(s, spec, hyp, (DELTA,)) for s in SYSTEMS)
        ev = evaluate(res, hs=(DELTA,)).iloc[0].to_dict()
        elig = [r for r in res if lab.loc[r["system"], "role"] in ("fault", "control")]
        a = auc([r.get("end_over_baseline") for r in res if lab.loc[r["system"], "role"] == "fault"], [r.get("end_over_baseline") for r in res if lab.loc[r["system"], "role"] == "control"])
        rows.append(dict(scenario=name, systems_with_baseline=sum(r["baseline"] for r in elig), systems_evaluated=len(elig), median_observations=float(np.median([r["n_obs"] for r in elig])),
                         auc_end_over_baseline=a, **{k: ev[k] for k in ("detected", "n_fault_systems", "early_flags_on_fault_systems", "control_false_flags", "n_control_systems", "delay_median_days")}))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "minimum_data_table_lfp.csv", index=False)


if __name__ == "__main__":
    main()
