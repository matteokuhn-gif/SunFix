"""Phase 6a: cost-aware evaluation on realistic-prevalence landmarks (all system-days), Bboxx lead-acid data.

ALL COST VALUES ARE HYPOTHETICAL (config/costs.yaml). Outputs are mechanics and shapes, not findings.

Design
- Landmarks: every system-day at age >= 120 d (sunfix.landmarks). Label y = fails within 30 d; alive batteries' last 30 d dropped (censored).
- Split by battery, by ACTIVATION DATE (never by row): earliest 50% train, next 20% calibration, latest 30% test.
  No location metadata exist, so the leave-one-district-out split cannot be done.
- Model: gradient boosting on resistance + usage features at the landmark day; Platt (and isotonic, reported) calibration on the calibration cohort.
- Policy: alert when calibrated score >= tau for N consecutive days; optional remote confirmation; capacity K dispatches per 1,000 systems per month;
  cooldown after an alert. An alert is TRUE if the battery's repair follows within W days (config useful_window_days).
  Alive batteries' last W days are excluded from exposure (outcome unknown).
- Expected-value accounting (no random draws): confirmation turns an alert into a dispatch with probability s_c (true) or f_c (false);
  capacity scales dispatch probability down proportionally within each 30-day calendar bucket. A true alert that is not dispatched becomes a
  missed failure (no second chance: conservative).
- Operating point (tau, N) is chosen on the CALIBRATION cohort and reported on the TEST cohort. Never tuned on test.
- 'mult' re-weights failed batteries to emulate a lower/higher base rate (1.0 = as sampled; the sample is about 48% failed batteries, so the
  as-sampled hazard overstates a real fleet).
"""
import itertools

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .features import RES_CORE, STRESS_COLS
from .load import ROOT
from .landmarks import MIN_AGE

FEATS = RES_CORE + STRESS_COLS
SEED = 0
TAUS = np.geomspace(0.01, 0.9, 22)
NS = (1, 3, 5, 7)
RES = ROOT / "results"
FIG = RES / "figures"


def costs() -> dict:
    return yaml.safe_load(open(ROOT / "config" / "costs.yaml"))


def derived(c: dict, scale: dict | None = None) -> dict:
    c = dict(c)
    for k, v in (scale or {}).items():
        c[k] = c[k] * v
    c["C_FN"] = c["C_batt_emergency"] + c["C_visit"] + c["C_outage_per_day"] * c["D_outage_days"] + c["C_churn"] * c["p_churn_after_outage"]
    c["C_TP"] = c["C_visit"] + c["C_batt_planned"]
    return c


# ------------------------------------------------------------------ data, split, model, calibration
def split(df):
    b = df.drop_duplicates("ID").sort_values("activation")
    ids = b.ID.to_numpy(); n = len(ids)
    return dict(train=ids[: int(0.5 * n)], calib=ids[int(0.5 * n): int(0.7 * n)], test=ids[int(0.7 * n):])


def ece(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-12
    return float(sum(abs(p[(p >= lo) & (p < hi)].mean() - y[(p >= lo) & (p < hi)].mean()) * ((p >= lo) & (p < hi)).mean()
                     for lo, hi in zip(q[:-1], q[1:]) if ((p >= lo) & (p < hi)).any()))


def fit_scores(df, sp, H=30):
    tr = df[df.ID.isin(sp["train"]) & df[f"valid{H}"]]
    mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED).fit(tr[FEATS], tr[f"y{H}"])
    raw = mdl.predict_proba(df[FEATS])[:, 1]
    cal = df.ID.isin(sp["calib"]).to_numpy() & df[f"valid{H}"].to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6).fit(lg(raw[cal])[:, None], df[f"y{H}"].to_numpy()[cal])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[cal], df[f"y{H}"].to_numpy()[cal])
    out = df[["ID", "day", "failed", "end_day", "activation", f"y{H}", f"valid{H}"]].copy()
    out["raw"], out["platt"], out["iso"] = raw, platt.predict_proba(lg(raw)[:, None])[:, 1], iso.predict(raw)
    return out


def calibration_table(sc, sp, H=30):
    rows = []
    for name in ("test", "calib"):
        s = sc[sc.ID.isin(sp[name]) & sc[f"valid{H}"]]
        y = s[f"y{H}"].to_numpy()
        for col in ("raw", "platt", "iso"):
            p = s[col].to_numpy()
            rows.append(dict(cohort=name, scores=col, n=len(y), base_rate=y.mean(), mean_pred=p.mean(), brier=float(np.mean((p - y) ** 2)), ece=ece(p, y)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ simulation
class Cohort:
    """Per-battery daily calibrated scores and outcome info for one cohort of batteries."""

    def __init__(self, sc, ids, col, W):
        self.W = W
        self.b = []
        for i, g in sc[sc.ID.isin(ids)].groupby("ID", sort=True):
            g = g.sort_values("day")
            failed = bool(g.failed.iloc[0]); end = float(g.end_day.iloc[0])
            exp_end = end if failed else end - W
            g = g[g.day <= exp_end]
            if len(g) < 2:
                continue
            self.b.append(dict(id=i, failed=failed, fail_day=end if failed else np.inf, days=g.day.to_numpy(), s=g[col].to_numpy(),
                               act=float(g.activation.iloc[0]), months=(g.day.iloc[-1] - MIN_AGE + 1) / 30.0))
        self.n = len(self.b)
        self.failed = np.array([x["failed"] for x in self.b])
        self.months = np.array([x["months"] for x in self.b])
        cal = [np.floor((x["act"] / 86400.0 + x["days"]) / 30.0).astype(int) for x in self.b]
        self.nb = int(max(c.max() for c in cal) - min(c.min() for c in cal) + 1); b0 = min(c.min() for c in cal)
        self.cal = [c - b0 for c in cal]
        self.sysdays = np.zeros((self.n, self.nb))
        for j, c in enumerate(self.cal):
            np.add.at(self.sysdays[j], c, 1.0)
        self._ev = {}

    def events(self, tau, N, cooldown):
        key = (round(float(tau), 6), N, cooldown)
        if key in self._ev:
            return self._ev[key]
        bidx, bucket, score, tp, lead, flips = [], [], [], [], [], np.zeros(self.n)
        for j, x in enumerate(self.b):
            above = x["s"] >= tau
            if N > 1:
                c = np.r_[0, np.cumsum(above)]
                qual = (c[N:] - c[:-N]) == N
                qual = np.r_[np.zeros(N - 1, bool), qual]
            else:
                qual = above
            flips[j] = np.abs(np.diff(qual.astype(int))).sum()
            idx = np.flatnonzero(qual)
            days = x["days"]
            while len(idx):
                k = idx[0]
                is_tp = x["failed"] and (x["fail_day"] - days[k]) <= self.W
                bidx.append(j); bucket.append(self.cal[j][k]); score.append(x["s"][k]); tp.append(is_tp)
                lead.append(x["fail_day"] - days[k] if is_tp else np.nan)
                if is_tp:
                    break
                idx = idx[np.searchsorted(days[idx], days[k] + cooldown):]
        r = (np.array(bidx, int), np.array(bucket, int), np.array(score), np.array(tp, bool), np.array(lead), flips)
        self._ev[key] = r
        return r


def wquant(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)])


def evaluate(co: Cohort, tau, N, confirm, K, c: dict, w=None, mult=1.0):
    """Expected-value outcome per 1,000 systems per month."""
    w = np.ones(co.n) if w is None else w.astype(float)
    w = w * np.where(co.failed, mult, 1.0)
    bidx, bucket, score, tp, lead, flips = co.events(tau, N, c["cooldown_days"])
    we = w[bidx]
    pdisp = np.ones(len(bidx))
    if confirm:
        pdisp = np.where(tp, c["confirm"]["sensitivity"], c["confirm"]["false_confirm"])
    eff = pdisp.copy()
    if K is not None and len(bidx):
        cap = K / 1000.0 * (w @ co.sysdays) / 30.0
        exp = np.bincount(bucket, we * pdisp, minlength=co.nb)
        f = np.where(exp > 0, np.minimum(1.0, cap / np.maximum(exp, 1e-12)), 1.0)
        eff = pdisp * f[bucket]
    tp_w = we[tp] * eff[tp]
    fp_w = we[~tp] * eff[~tp]
    got = np.zeros(co.n, bool); got[bidx[tp]] = True
    n_fail = float(w[co.failed].sum())
    miss = n_fail - tp_w.sum()
    cost = tp_w.sum() * c["C_TP"] + miss * c["C_FN"] + fp_w.sum() * c["C_visit"] + (we.sum() * c["C_check"] if confirm else 0.0)
    mon = float((w * co.months).sum()) / 1000.0
    ld = lead[tp]
    r = dict(EC_per_1000_month=cost / mon, false_dispatch_per_1000_month=fp_w.sum() / mon, missed_per_1000_month=miss / mon,
             true_dispatch_per_1000_month=tp_w.sum() / mon, alerts_per_1000_month=we.sum() / mon, failures_per_1000_month=n_fail / mon,
             precision=tp_w.sum() / max(tp_w.sum() + fp_w.sum(), 1e-12), recall=tp_w.sum() / max(n_fail, 1e-12),
             flips_per_system_month=float((w * flips).sum() / (w * co.months).sum()))
    if len(ld) and tp_w.sum() > 0:
        r.update(lead_median=wquant(ld, tp_w, 0.5), lead_p10=wquant(ld, tp_w, 0.10), lead_share_ge30=float(tp_w[ld >= 30].sum() / tp_w.sum()))
    else:
        r.update(lead_median=np.nan, lead_p10=np.nan, lead_share_ge30=np.nan)
    return r


def reactive(co: Cohort, c, w=None, mult=1.0):
    w = (np.ones(co.n) if w is None else w) * np.where(co.failed, mult, 1.0)
    mon = float((w * co.months).sum()) / 1000.0
    nf = float(w[co.failed].sum())
    return dict(EC_per_1000_month=nf * c["C_FN"] / mon, false_dispatch_per_1000_month=0.0, missed_per_1000_month=nf / mon,
                true_dispatch_per_1000_month=0.0, failures_per_1000_month=nf / mon, precision=np.nan, recall=0.0)


def age_based(co: Cohort, c, A, w=None, mult=1.0):
    """Replace every battery at age A; failures before A are reactive."""
    w = (np.ones(co.n) if w is None else w) * np.where(co.failed, mult, 1.0)
    fail_day = np.array([x["fail_day"] for x in co.b]); last = np.array([x["days"][-1] for x in co.b])
    react = co.failed & (fail_day < A)
    planned = ~react & (last >= A)
    months = np.maximum(np.minimum(last, A) - MIN_AGE + 1, 0) / 30.0
    mon = float((w * months).sum()) / 1000.0
    cost = (w * react).sum() * c["C_FN"] + (w * planned).sum() * c["C_TP"]
    return dict(EC_per_1000_month=cost / mon, false_dispatch_per_1000_month=float((w * planned).sum() / mon),
                missed_per_1000_month=float((w * react).sum() / mon), true_dispatch_per_1000_month=0.0, precision=np.nan, recall=np.nan)


def grid(co, c, mult=1.0):
    rows = []
    for tau, N, conf, K in itertools.product(TAUS, NS, (False, True), c["K_per_1000_month"]):
        r = evaluate(co, tau, N, conf, K, c, mult=mult)
        r.update(tau=float(tau), N=N, confirm=conf, K=-1 if K is None else K)
        rows.append(r)
    return pd.DataFrame(rows)


def best_per_config(g):
    return g.loc[g.groupby(["N", "confirm", "K"]).EC_per_1000_month.idxmin()]


def bootstrap(co, tau, N, conf, K, c, n=200, mult=1.0):
    rng = np.random.default_rng(SEED)
    rows = [evaluate(co, tau, N, conf, None if K == -1 else K, c, w=rng.multinomial(co.n, np.ones(co.n) / co.n), mult=mult) for _ in range(n)]
    d = pd.DataFrame(rows)
    return {f"{k}_lo": d[k].quantile(0.025) for k in d} | {f"{k}_hi": d[k].quantile(0.975) for k in d}
