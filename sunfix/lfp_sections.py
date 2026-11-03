"""Section start/end timestamps used by BattGP's per-cell run (paper settings), so pack-only results share the same time axis.
Output: results/lfp_section_times.csv. Run: python -m sunfix.lfp_sections (about 5 min)."""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "BattGP"))
from src import config as cfg
cfg.PATH_FIELDDATA_DATA = str(ROOT / "data" / "battgp_lfp" / "field_data.zip"); cfg.PATH_DATA_CACHE = str(ROOT / "data" / "battgp_cache")
cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = str(ROOT / "external" / "BattGP" / "data" / "ocv_linear_approx.csv")
from src.batt_data import batt_data, data_utils
from src.batt_data.batt_data import SegmentCriteria
warnings.filterwarnings("ignore")


def main():
    ocv = data_utils.read_cell_characteristics(path=cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC)
    rows = []
    for sid in range(1, 22):
        bd = batt_data.BattData(str(sid), ocv, segment_selection=True, segment_criteria=SegmentCriteria(soc_upper_limit=94, soc_lower_limit=cfg.SOC_LOWER_LIMIT, ibat_upper_limit=cfg.Ibat_UPPER_LIMIT,
                                ibat_lower_limit=-200, t_upper_limit=cfg.T_UPPER_LIMIT, t_lower_limit=cfg.T_LOWER_LIMIT), gap_removal=cfg.GAP_REMOVAL, min_data_threshold=2000)
        idx = pd.to_datetime(bd.df.index) if not isinstance(bd.df.index, pd.DatetimeIndex) else bd.df.index
        rows.append(dict(system=sid, start=idx.min(), end=idx.max(), span_days=(idx.max() - idx.min()).total_seconds() / 86400, n_rows=len(bd.df)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "lfp_section_times.csv", index=False)


if __name__ == "__main__":
    main()
