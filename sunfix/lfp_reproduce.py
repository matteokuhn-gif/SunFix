"""Phase 6d.1: reproduce BattGP's per-cell fault probabilities on the Zenodo LFP field data (systems given on the command line).
Uses BattGP unchanged (spatio-temporal GP, CPU, shipped LFP hyperparameters, r0_band 0.55 mOhm, upper threshold 2 mOhm, as in gp_runner.py).
Data is read from the downloaded zip (read-only); BattGP's cache goes to data/battgp_cache. One system at a time (memory).
Run: python -m sunfix.lfp_reproduce 3 9 6
"""
import gc
import json
import sys
import time
import resource
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "BattGP"))
from src import config as cfg   # noqa: E402

cfg.PATH_FIELDDATA_DATA = str(ROOT / "data" / "battgp_lfp" / "field_data.zip")
cfg.PATH_DATA_CACHE = str(ROOT / "data" / "battgp_cache")
cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = str(ROOT / "external" / "BattGP" / "data" / "ocv_linear_approx.csv")
cfg.PATH_RESULTS = str(ROOT / "data" / "battgp_results") + "/"
cfg.PATH_FIGURES_DATA_VIS = str(ROOT / "data" / "battgp_results" / "figs") + "/"

from src.batt_data import batt_data, data_utils            # noqa: E402
from src.batt_data.batt_data import SegmentCriteria         # noqa: F401,E402
from src.batt_models.battgp_spatiotemporal import BattGP_SpatioTemporal   # noqa: E402
from src.batt_models.fault_probabilities import calc_fault_probabilities   # noqa: E402
from src.batt_models.ref_strategy import RefStrategy        # noqa: E402
from src.operating_point import Op                           # noqa: E402
from src.path_setup import setup_paths                       # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "data" / "battgp_results"
R0_BAND, R0_UPPER = 0.55e-3, 2.0e-3
OP = Op(-15, 90, 25)


def run(sid: str):
    t0 = time.time()
    ocv = data_utils.read_cell_characteristics(path=cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC)
    battdata = batt_data.BattData(sid, ocv, segment_selection=True, segment_criteria=SegmentCriteria(soc_upper_limit=cfg.SOC_UPPER_LIMIT, soc_lower_limit=cfg.SOC_LOWER_LIMIT,
                                                                       ibat_upper_limit=cfg.Ibat_UPPER_LIMIT, ibat_lower_limit=cfg.Ibat_LOWER_LIMIT,
                                                                       t_upper_limit=cfg.T_UPPER_LIMIT, t_lower_limit=cfg.T_LOWER_LIMIT),
                                  gap_removal=cfg.GAP_REMOVAL, min_data_threshold=2000)
    t1 = time.time()
    model = BattGP_SpatioTemporal(battdata, max_age=None, device=torch.device("cpu"), sampling_time_sec=3600, max_batch_size=1000,
                                  save_path=str(OUT), ref_strategy=RefStrategy(OP), basis_vector_strategy="kmeans", nbasis=[60])
    res = model.predict_cell_r0_op(smooth=True)
    t2 = time.time()
    out = {}
    for causal in (False, True):
        df = calc_fault_probabilities(res, causal=causal, r0_band=R0_BAND, r0_upper_threshold=R0_UPPER)
        out["causal" if causal else "smoothed"] = df
    OUT.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_parquet(OUT / f"fault_probs_sys{sid}_{k}.parquet")
    rows = []
    for k, df in out.items():
        for c in df.columns:
            v = df[c].to_numpy(float)
            rows.append(dict(system=sid, variant=k, column=str(c), max_prob=float(np.nanmax(v)), final_prob=float(v[~np.isnan(v)][-1]) if (~np.isnan(v)).any() else np.nan,
                             frac_time_above_0_5=float(np.nanmean(v > 0.5))))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / f"fault_summary_sys{sid}.csv", index=False)
    info = dict(system=sid, rows_loaded=int(len(battdata.df)), load_s=t1 - t0, gp_s=t2 - t1, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
    print(json.dumps(info), flush=True)
    print(s.pivot_table(index="column", columns="variant", values="max_prob").round(3).to_string(), flush=True)
    del battdata, model, res, out; gc.collect()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    setup_paths()
    for sid in sys.argv[1:]:
        run(sid)
