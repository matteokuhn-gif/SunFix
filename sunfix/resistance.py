"""Time-varying battery resistance with BattGP's recursive spatiotemporal GP.

Model (Aitio & Howey Eq. 3-5, BattGP implementation): R0(t, x) = Wiener-velocity(t) + SE(x), x = operating point.
Observations are R_obs = (V - OCV) / |I| with per-row noise (sigma_v^2 / I^2).
BattGP's ApproxSpatioTemporalGP is reused unchanged; we add heteroskedastic noise and likelihood bookkeeping.
Only the causal filter is run (no smoothing), so the output at time t uses data up to t.
"""
import sys
from dataclasses import dataclass, field

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from .load import ROOT, load_meta
from .clean import segments_for
from .soc import add_soc, usable_rows

sys.path.insert(0, str(ROOT / "external" / "BattGP"))
from src.gp.spatiotemporal_gp import ApproxSpatioTemporalGP  # noqa: E402
from src.gp.wiener_kernel_temporal import WienerTemporalKernel  # noqa: E402

torch.set_default_dtype(torch.float64)
T_NORM = 400.0 * 86400.0   # s, Aitio's nominal 400-day life
ROW_STRIDE = 5             # keep every 5th 1-min row (5-min grid) for speed
# Aitio's population medians (supplement Fig. 5) as starting values: sigma_f0, sigma_f1, then lengthscales
START = dict(sf0=2.26, sf1=0.34, ls=(1.94, 2.79, 3.96))    # lengthscales for (I, SOC, T)
VARIANTS = {
    "ocv": dict(dims=("I", "soc", "T"), note="Coulomb-counted SOC + GITT OCV curve (Aitio-style)"),
    "nocv": dict(dims=("I", "T"), note="no OCV curve: OCV approximated by the rest voltage before the segment"),
}


@dataclass
class Prepared:
    t: np.ndarray            # normalised time of each row
    seg: np.ndarray
    X: np.ndarray            # raw operating point columns (|I|, [soc], T)
    y: np.ndarray            # R_obs, Ohm
    noise: np.ndarray        # variance of y
    v: np.ndarray
    i: np.ndarray
    base_v: np.ndarray       # OCV (or rest voltage) used, V


def prepare(batt_id: int, variant: str, meta: pd.DataFrame | None = None, stride: int = ROW_STRIDE,
            max_t: float | None = None) -> Prepared | None:
    """Rows for the resistance model. max_t (unix s) cuts the data, used to guarantee no look-ahead."""
    m = (meta if meta is not None else load_meta()).loc[batt_id]
    seg = add_soc(segments_for(batt_id, meta))
    if max_t is not None:
        seg = seg[seg.t <= max_t]
    if variant == "ocv":
        u = usable_rows(seg)
        base, noise_v = u.ocv.to_numpy(), u.noise_v2.to_numpy()
        cols = ["I", "soc", "T"]
    else:
        u = seg[(seg.I < -0.2) & (seg.v_start < seg.V)].reset_index(drop=True)
        base, noise_v = u.v_start.to_numpy(), np.full(len(u), 0.15**2)
        cols = ["I", "T"]
    if len(u) == 0:
        return None
    u = u.iloc[::stride]
    base, noise_v = base[::stride], noise_v[::stride]
    ic = -u.I.to_numpy()
    X = u[cols].to_numpy().copy()
    X[:, 0] = ic
    return Prepared(t=(u.t.to_numpy() - m.activation_time.timestamp()) / T_NORM, seg=u.seg.to_numpy(), X=X,
                    y=(u.V.to_numpy() - base) / ic, noise=noise_v / ic**2, v=u.V.to_numpy(), i=u.I.to_numpy(), base_v=base)


@dataclass
class Normaliser:
    mean: np.ndarray
    std: np.ndarray
    basis: np.ndarray

    def __call__(self, X):
        return (X - self.mean) / self.std


def fit_normaliser(preps: list[Prepared], n_basis: int = 20, seed: int = 0) -> Normaliser:
    """Population moments and shared k-means basis vectors (Aitio: 20 points) from the training batteries."""
    X = np.vstack([p.X for p in preps])
    mean, std = X.mean(0), X.std(0)
    Z = (X - mean) / std
    rng = np.random.default_rng(seed)
    Z = Z[rng.choice(len(Z), min(len(Z), 50000), replace=False)]
    basis = KMeans(n_clusters=n_basis, n_init=3, random_state=seed).fit(Z).cluster_centers_
    basis = np.vstack([basis, np.zeros((1, X.shape[1]))])   # reference operating point = population mean
    return Normaliser(mean, std, basis)


class HeteroGP(ApproxSpatioTemporalGP):
    """BattGP's recursive GP with per-row noise variances and innovation log-likelihood."""

    def update_h(self, Xs, y, noise, tau2=0.0):
        H, Ktt = self._get_output_matrix(Xs)
        C = self._get_covariance(H, Ktt, self.P)
        v = y.reshape((-1, 1)) - H @ self.z
        S = C + np.diag(noise) + tau2   # tau2: variance of a per-segment random offset shared by all rows (extension)
        L = np.linalg.cholesky(S + 1e-12 * np.eye(len(S)))
        a = np.linalg.solve(L, v)
        ll = -0.5 * float(a.T @ a) - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi)
        K = self.P @ H.T @ np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(S))))
        self.z = self.z + K @ v
        self.P = self.P - K @ H @ self.P
        return ll, v.ravel(), np.diag(S)


def build_gp(norm: Normaliser, sf0: float, sf1: float, ls: tuple) -> HeteroGP:
    d = norm.basis.shape[1]
    k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
    k.outputscale = sf1**2
    k.base_kernel.lengthscale = torch.tensor([list(ls)[:d]], dtype=torch.float64)
    return HeteroGP(torch.tensor(norm.basis), k, WienerTemporalKernel(sf0**2), noise_var=0.0, smoothing_steps=None,
                    device=torch.device("cpu"))


def run_battery(p: Prepared, norm: Normaliser, sf0: float, sf1: float, ls: tuple, slices: bool = False, tau: float = 0.0):
    """Causal filter over a battery's segments. Returns (per-segment DataFrame, total log-lik, slices dict|None)."""
    gp = build_gp(norm, sf0, sf1, ls)
    Z = norm(p.X)
    ref = np.zeros((1, Z.shape[1]))
    rows, total_ll, t_prev = [], 0.0, 0.0
    bounds = np.flatnonzero(np.r_[True, np.diff(p.seg) != 0, True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        t_seg = p.t[a:b]
        if t_seg[0] > t_prev:
            gp.time_step(float(t_seg[0] - t_prev))
        ll, v, S = gp.update_h(Z[a:b], p.y[a:b], p.noise[a:b], tau**2)
        total_ll += ll
        t_prev = float(t_seg[-1])
        r, rv = gp.predict(ref)
        z0, z1 = float(gp.z[0, 0]), float(gp.z[1, 0])
        sl = gp.P[:2, :2]
        i_abs = p.X[a:b, 0]
        rows.append(dict(t=t_prev, n=b - a, r0=float(r[0]), r0var=float(rv[0]), z0=z0, z1=z1, p00=sl[0, 0], p01=sl[0, 1],
                         p11=sl[1, 1], r_op=float(r[0]) - z0, innov_v_rmse=float(np.sqrt(np.mean((v * i_abs) ** 2))),
                         innov_z=float(np.mean(v**2 / S))))
    out = pd.DataFrame(rows)
    sl_out = None
    if slices:   # operating-point slices of the final model, other inputs at the reference point
        sl_out = {}
        q = np.linspace(5, 95, 21)
        for j, name in enumerate(("I", "soc", "T")[: Z.shape[1]] if Z.shape[1] == 3 else ("I", "T")):
            grid = np.zeros((len(q), Z.shape[1])); grid[:, j] = np.percentile(Z[:, j], q)
            mu, var = gp.predict(grid)
            sl_out[name] = dict(x=grid[:, j] * norm.std[j] + norm.mean[j], r=mu - gp.z[0, 0], sd=np.sqrt(np.maximum(var, 0)))
    return out, total_ll, sl_out
