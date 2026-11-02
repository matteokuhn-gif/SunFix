"""Cleaning and Aitio & Howey (2021) charging-segment selection.

Current convention in the raw data: negative = charging, positive = discharging.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .load import ROOT, cached_raw, load_meta

SEG_DIR = ROOT / "data" / "segments_v2"

V_MIN, V_MAX = 9.0, 16.0   # physically impossible outside this for a 12 V VRLA (observed 0 V and up to 20 V)
CHG_I = -0.05              # A, below this the battery counts as charging
SEG = dict(min_dur=6000.0, v_start=(11.5, 12.9), i_start=0.1, v_max_min=14.0, max_gap=610.0)  # Aitio Table 1
V_TRUNC = 14.0             # segments truncated at 14 V (Aitio)
MIN_I_ABS = 0.2            # A, rows with |I| below this dropped for conditioning (Aitio)
GRID = 60.0                # s, 1-min interpolation grid (Aitio)


def clean_raw(batt_id: int, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw cleaning. Returns columns t, I, V, T for the battery's in-service life.

    1. keep activation <= t <= repair date (failed) or end of data (alive); post-repair data is
       sparse and does not belong to the failed battery
    2. drop NaN, drop V outside [9, 16] V, drop duplicate timestamps (keep first), sort by time
    """
    m = (meta if meta is not None else load_meta()).loc[batt_id]
    d = cached_raw(batt_id)
    t0 = m.activation_time.timestamp()
    t1 = m.failure_time.timestamp() if m.failed else np.inf
    d = d[(d.t >= t0) & (d.t <= t1)]
    d = d.dropna()
    d = d[(d.V >= V_MIN) & (d.V <= V_MAX)]
    d = d.drop_duplicates("t", keep="first").sort_values("t")
    return d.reset_index(drop=True)


def charging_segments(d: pd.DataFrame, criteria: dict = SEG) -> list[tuple[int, int]]:
    """Index ranges [a, b) of qualifying charging segments in a cleaned frame.

    A segment is a maximal run of samples with I < CHG_I and no time gap > max_gap. It qualifies if
    duration > 6000 s, the sample before the run (rest state) has V in 11.5-12.9 V and |I| < 0.1 A,
    and max V in the run > 14 V.
    """
    t, I, V = d.t.to_numpy(), d.I.to_numpy(), d.V.to_numpy()
    chg = I < CHG_I
    gap = np.r_[False, np.diff(t) > criteria["max_gap"]]
    new = chg & (~np.r_[False, chg[:-1]] | gap)
    starts = np.flatnonzero(new)
    ends = np.flatnonzero(chg & ~np.r_[chg[1:], False] | (chg & np.r_[gap[1:], False])) + 1
    out = []
    for a, b in zip(starts, ends):
        if a == 0 or b - a < 2:
            continue
        if t[b - 1] - t[a] <= criteria["min_dur"]:
            continue
        if not (criteria["v_start"][0] <= V[a - 1] <= criteria["v_start"][1]) or abs(I[a - 1]) >= criteria["i_start"]:
            continue
        if V[a:b].max() <= criteria["v_max_min"]:
            continue
        out.append((a, b))
    return out


def interpolate_segment(d: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """Truncate at first V > 14 V and PCHIP-interpolate to a 1-min grid. The |I| < 0.2 A filter is applied
    later (sunfix.soc) because Coulomb counting needs the low-current rows. Adds v_start, the rest voltage
    before the segment."""
    s = d.iloc[a:b]
    over = np.flatnonzero(s.V.to_numpy() > V_TRUNC)
    if len(over):
        s = s.iloc[: over[0]]
    if len(s) < 3:
        return s.iloc[:0]
    t = s.t.to_numpy()
    g = np.arange(np.ceil(t[0] / GRID) * GRID, t[-1], GRID)
    if len(g) < 2:
        return s.iloc[:0]
    out = pd.DataFrame({"t": g})
    for c in ("I", "V", "T"):
        out[c] = PchipInterpolator(t, s[c].to_numpy())(g)
    out["v_start"] = d.V.iloc[a - 1]
    return out


def segments_for(batt_id: int, meta: pd.DataFrame | None = None, cache: bool = True) -> pd.DataFrame:
    """All qualifying, interpolated charging segments of a battery with a `seg` column."""
    path = SEG_DIR / f"{batt_id}.parquet"  # v2: unfiltered grid + v_start
    if cache and path.exists():
        return pd.read_parquet(path)
    d = clean_raw(batt_id, meta)
    parts = []
    for k, (a, b) in enumerate(charging_segments(d)):
        s = interpolate_segment(d, a, b)
        if len(s):
            parts.append(s.assign(seg=k))
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "I", "V", "T", "v_start", "seg"])
    if cache:
        SEG_DIR.mkdir(parents=True, exist_ok=True)
        res.to_parquet(path, index=False)
    return res
