"""Cache raw Parquet and Aitio-style segments for all 1,027 batteries (parallel, ~3 GB on disk)."""
from joblib import Parallel, delayed

from .clean import segments_for
from .load import load_meta, zip_index


def one(i, m):
    return i, int(segments_for(i, m).seg.nunique())


if __name__ == "__main__":
    m = load_meta()
    res = Parallel(n_jobs=6)(delayed(one)(i, m) for i in sorted(zip_index()))
    import pandas as pd
    pd.DataFrame(res, columns=["ID", "n_seg"]).to_csv("data/segments_count_all.csv", index=False)
    print("done", len(res))
