"""Open-circuit-voltage curve of the Bboxx 12 V / 20 Ah VRLA from the GITT discharge in GITT_OCV.mpt."""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .load import ROOT

T_DISCHARGE = (67350.0, 367000.0)  # s, the discharge GITT (Ns 4-6); the file also holds two charge segments


def gitt_points() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "GITT_OCV.mpt", sep="\t", usecols=["time_s", "Ewe_V", "I_mA"])
    g = g[(g.time_s >= T_DISCHARGE[0]) & (g.time_s <= T_DISCHARGE[1])].reset_index(drop=True)
    t, I, V = g.time_s.to_numpy(), g.I_mA.to_numpy() / 1000.0, g.Ewe_V.to_numpy()   # I in A, negative = discharge
    q = np.r_[0.0, np.cumsum(-0.5 * (I[1:] + I[:-1]) * np.diff(t))] / 3600.0         # Ah discharged
    rest = np.abs(I) < 0.005
    end_of_rest = np.flatnonzero(rest[:-1] & ~rest[1:])                                # last rest sample before a pulse
    idx = np.r_[0, end_of_rest]
    return pd.DataFrame({"q_dis_Ah": q[idx], "ocv_V": V[idx]})


class OcvCurve:
    """OCV(soc) with soc = Ah of charge remaining relative to the full state, 0 (empty) to Q_FULL (full)."""

    def __init__(self):
        p = gitt_points()
        p = p.groupby(p.ocv_V.round(4)).first().reset_index(drop=True).sort_values("q_dis_Ah")
        p["ocv_V"] = np.minimum.accumulate(p.ocv_V.to_numpy())   # enforce monotone decrease with discharge
        self.q_full = float(p.q_dis_Ah.max())
        self.points = p
        soc = self.q_full - p.q_dis_Ah.to_numpy()                 # ascending after reversal
        v = p.ocv_V.to_numpy()
        order = np.argsort(soc)
        self._f = PchipInterpolator(soc[order], v[order], extrapolate=True)
        self._inv = PchipInterpolator(v[order][np.argsort(v[order])], soc[order][np.argsort(v[order])], extrapolate=True)

    def ocv(self, soc):
        return self._f(np.clip(soc, 0.0, self.q_full))

    def docv(self, soc):
        return self._f.derivative()(np.clip(soc, 0.0, self.q_full))

    def soc_from_ocv(self, v):
        return np.clip(self._inv(v), 0.0, self.q_full)


if __name__ == "__main__":
    c = OcvCurve()
    print(c.points.round(3).to_string()); print("q_full", c.q_full)
