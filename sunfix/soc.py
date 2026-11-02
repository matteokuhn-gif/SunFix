"""State of charge by Coulomb counting within a charging segment, following Aitio & Howey Eq. 1 and 6.

SOC is in Ah of charge above empty (0 = empty, 21.9 = full, from the GITT curve), not acid molarity. They differ by a
constant factor (F * V_electrolyte), and inputs are normalised, so the GP is unaffected.
"""
import numpy as np
import pandas as pd

from .clean import MIN_I_ABS
from .ocv import OcvCurve

# Gassing current I_gas = I0 * exp(cT (T - T0) + cV (V - V0)). Aitio take these from literature without listing them;
# these are generic VRLA values and are ASSUMPTIONS (see docs/phase2_report.md).
GAS = dict(I0=0.005, T0=25.0, cT=0.07, V0=13.5, cV=8.0)
SIGMA_V = 0.15       # V, voltage measurement plus OCV uncertainty (Aitio Eq. 6A)
REL_DSOC = 0.10      # relative uncertainty of d(soc)/dt (Aitio Eq. 6B)
_OCV = OcvCurve()


def add_soc(seg: pd.DataFrame) -> pd.DataFrame:
    """Per-segment SOC, SOC variance and OCV. Input is the unfiltered 1-min grid from clean.segments_for."""
    out = []
    for _, s in seg.groupby("seg", sort=False):
        s = s.copy()
        dt = np.diff(s.t.to_numpy(), prepend=s.t.iloc[0]) / 3600.0
        gas = GAS["I0"] * np.exp(GAS["cT"] * (s["T"] - GAS["T0"]) + GAS["cV"] * (s.V - GAS["V0"]))
        net = np.maximum(-s.I.to_numpy() - np.minimum(gas.to_numpy(), -s.I.to_numpy()), 0.0)  # A into the active material
        dsoc = net * dt
        soc0 = float(_OCV.soc_from_ocv(s.v_start.iloc[0]))
        s["soc"] = np.minimum(soc0 + np.cumsum(dsoc), _OCV.q_full)
        s["soc_var"] = np.cumsum((REL_DSOC * dsoc) ** 2)
        s["ocv"] = _OCV.ocv(s.soc.to_numpy())
        s["noise_v2"] = SIGMA_V**2 + s.soc_var * _OCV.docv(s.soc.to_numpy()) ** 2
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else seg.assign(soc=[], soc_var=[], ocv=[], noise_v2=[])


def usable_rows(s: pd.DataFrame) -> pd.DataFrame:
    """Aitio's row filters: |I| >= 0.2 A while charging, and estimated OCV not above terminal voltage."""
    return s[(s.I < -MIN_I_ABS) & (s.ocv < s.V)].reset_index(drop=True)
