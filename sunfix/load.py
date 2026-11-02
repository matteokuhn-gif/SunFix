"""Stream batteries out of the Aitio & Howey zips without extracting them, cache as Parquet."""
import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
SEED = 0


@lru_cache(maxsize=1)
def zip_index() -> dict[int, tuple[Path, str]]:
    """Map battery ID -> (zip path, member name). Reads zip directories only."""
    index = {}
    for zp in sorted(ROOT.glob("set_*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                index[int(Path(name).stem)] = (zp, name)
    return index


def read_raw(batt_id: int) -> pd.DataFrame:
    """Raw telemetry as stored: columns t (unix s), I (A), V (V), T (degC)."""
    zp, name = zip_index()[batt_id]
    with zipfile.ZipFile(zp) as z:
        arr = np.load(io.BytesIO(z.read(name)))["arr_0"]
    return pd.DataFrame(arr, columns=["t", "I", "V", "T"])


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "meta_data.csv")
    m["failed"] = ~m["STILL_ALIVE"].astype(bool) if m["STILL_ALIVE"].dtype == bool else m["STILL_ALIVE"].astype(str).str.upper().ne("TRUE")
    m["activation_time"] = pd.to_datetime(m["ACTIVATED"], format="%m/%d/%Y", utc=True)
    m["failure_time"] = pd.to_datetime(m["IN_REPAIR_SYSTEM"], format="%m/%d/%Y", utc=True)
    return m.set_index("ID")


def dev_subset(n_each: int = 30, seed: int = SEED) -> list[int]:
    """Fixed-seed development subset: n_each failed + n_each alive battery IDs."""
    m = load_meta()
    rng = np.random.default_rng(seed)
    ids = []
    for flag in (True, False):
        pool = np.sort(m.index[m["failed"] == flag].to_numpy())
        ids += sorted(rng.choice(pool, n_each, replace=False).tolist())
    return ids


def cached_raw(batt_id: int) -> pd.DataFrame:
    """Raw telemetry via the Parquet cache (created on first use)."""
    path = PARQUET_DIR / f"{batt_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = read_raw(batt_id)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return df
