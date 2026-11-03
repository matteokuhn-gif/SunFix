"""Phase 6a runner: python -m sunfix.run_6a  (about 5-10 min). All costs hypothetical, see sunfix/eval_ops.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval_ops import *   # noqa: F401,F403
from .eval_ops import FIG, NS, RES, SEED, TAUS, Cohort, age_based, best_per_config, bootstrap, calibration_table, costs, derived, evaluate, fit_scores, grid, reactive, split
from .load import ROOT

W_DAYS = None


def best_cfg(co_cal, c, mult=1.0, K_only_unlimited=True):
    rows = []
    import itertools
    for tau, N, conf in itertools.product(TAUS, NS, (False, True)):
        r = evaluate(co_cal, tau, N, conf, None, c, mult=mult); r.update(tau=tau, N=N, confirm=conf); rows.append(r)
    d = pd.DataFrame(rows)
    return d.loc[d.EC_per_1000_month.idxmin()]


def main():
    raw = costs(); c0 = derived(raw); W = raw["useful_window_days"]
    df = pd.read_parquet(ROOT / "data" / "landmarks.parquet")
    sp = split(df)
    print({k: len(v) for k, v in sp.items()}, "failed share:", {k: round(float(df[df.ID.isin(v)].drop_duplicates('ID').failed.mean()), 3) for k, v in sp.items()})
    sc = fit_scores(df, sp)
    cal = calibration_table(sc, sp); cal.to_csv(RES / "calibration_bboxx.csv", index=False); print(cal.round(4).to_string(index=False))
    tr, ca, te = (Cohort(sc, sp[k], "platt", W) for k in ("train", "calib", "test"))
    print("cohorts (batteries, failed):", [(x.n, int(x.failed.sum())) for x in (tr, ca, te)])
    # hazard by age (empirical, all batteries, as sampled)
    v = df[df.valid30].assign(age_bin=lambda d: (d.day // 60) * 60)
    v.groupby("age_bin").agg(system_days=("y30", "size"), monthly_failure_hazard=("y30", "mean")).to_csv(RES / "hazard_by_age_bboxx.csv")
    # --- operating point chosen on calibration cohort, reported on test
    rows = []
    g_cal, g_te = grid(ca, c0), grid(te, c0)
    g_te.to_csv(ROOT / "data" / "grid_test_bboxx.csv", index=False)
    sel = best_per_config(g_cal)
    for _, s in sel.iterrows():
        K = None if s.K == -1 else int(s.K)
        r = evaluate(te, s.tau, int(s.N), bool(s.confirm), K, c0)
        r.update(bootstrap(te, s.tau, int(s.N), bool(s.confirm), s.K, c0, n=150))
        r.update(policy=f"tau={s.tau:.3f} N={int(s.N)} confirm={'on' if s.confirm else 'off'} K={'inf' if K is None else K}", tau=s.tau, N=int(s.N),
                 confirm=bool(s.confirm), K=s.K, kind="model_selected_on_calibration")
        rows.append(r)
    # baselines
    base = reactive(te, c0); base.update(policy="reactive only", kind="baseline"); rows.append(base)
    Agrid = list(range(200, 760, 20))
    A = min(Agrid, key=lambda a: age_based(tr, c0, a)["EC_per_1000_month"])
    ab = age_based(te, c0, A); ab.update(policy=f"age-based replacement at {A} d (A chosen on training cohort)", kind="baseline"); rows.append(ab)
    pstar = c0["C_visit"] / (c0["C_visit"] + c0["C_FN"] - c0["C_TP"])
    el = evaluate(te, pstar, 1, False, None, c0); el.update(policy=f"Elkan threshold p*={pstar:.3f}, N=1", kind="baseline", tau=pstar, N=1); rows.append(el)
    # Phase 3 style: threshold maximising balanced accuracy of the 30-day label on the calibration cohort, N=1
    s_ = sc[sc.ID.isin(sp["calib"]) & sc.valid30]; y = s_.y30.to_numpy(bool)
    ba = [(0.5 * (((s_.platt >= t) & y).sum() / y.sum() + ((s_.platt < t) & ~y).sum() / (~y).sum()), t) for t in TAUS]
    t3 = max(ba)[1]
    p3 = evaluate(te, t3, 1, False, None, c0); p3.update(policy=f"Phase-3-style (balanced-accuracy threshold {t3:.3f}), N=1", kind="phase3_reexpressed", tau=t3, N=1); rows.append(p3)
    out = pd.DataFrame(rows)
    out["cost_status"] = "HYPOTHETICAL costs: shapes not findings"
    lead = ["policy", "kind", "EC_per_1000_month", "false_dispatch_per_1000_month", "missed_per_1000_month", "lead_median"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(RES / "ops_metrics_bboxx.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out[lead + ["precision", "recall", "lead_p10", "lead_share_ge30", "flips_per_system_month"]].round(2).to_string(index=False))
    # --- sensitivity (K unlimited): re-select on calibration for every perturbation, report on test
    srows = []
    def one(label, scale, mult):
        c = derived(raw, scale); b = best_cfg(ca, c, mult)
        r = evaluate(te, b.tau, int(b.N), bool(b.confirm), None, c, mult=mult); rc = reactive(te, c, mult=mult)
        Ab = min(Agrid, key=lambda a: age_based(tr, c, a, mult=mult)["EC_per_1000_month"]); ag = age_based(te, c, Ab, mult=mult)
        return dict(param=label, scale=(scale or {}).get(label, mult if label == "base_rate" else 1.0), EC_policy=r["EC_per_1000_month"], EC_reactive=rc["EC_per_1000_month"],
                    EC_age=ag["EC_per_1000_month"], saving_vs_reactive=rc["EC_per_1000_month"] - r["EC_per_1000_month"],
                    best_tau=b.tau, best_N=int(b.N), confirm=bool(b.confirm), false_dispatch_per_1000_month=r["false_dispatch_per_1000_month"],
                    missed_per_1000_month=r["missed_per_1000_month"])
    srows.append(one("baseline", None, 1.0))
    for p in ("C_visit", "C_check", "C_batt_planned", "C_batt_emergency", "C_outage_per_day", "D_outage_days", "C_churn"):
        for f in (0.5, 1.5):
            srows.append(one(p, {p: f}, 1.0))
    for f in (0.25, 0.5, 1.5):
        srows.append(one("base_rate", None, f))
    pd.DataFrame(srows).to_csv(RES / "ops_sensitivity_bboxx.csv", index=False)
    print(pd.DataFrame(srows).round(2).to_string(index=False))
    # break-even surface: saving vs reactive over C_visit and base rate
    vs, bs = [0.25, 0.5, 1, 2, 3, 4], [0.05, 0.1, 0.25, 0.5, 1.0, 1.5]
    sav = np.array([[one("C_visit", {"C_visit": v_}, b_)["saving_vs_reactive"] for v_ in vs] for b_ in bs])
    pd.DataFrame(sav, index=[f"base_rate x{b}" for b in bs], columns=[f"C_visit x{v}" for v in vs]).to_csv(RES / "ops_breakeven_bboxx.csv")
    # --- figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a, conf in zip(ax, (False, True)):
        for N in NS:
            d = g_te[(g_te.N == N) & (g_te.confirm == conf) & (g_te.K == -1)].sort_values("tau")
            a.plot(d.tau, d.EC_per_1000_month, label=f"persistence N={N}")
        a.axhline(base["EC_per_1000_month"], color="k", ls="--", lw=1, label="reactive only"); a.axhline(ab["EC_per_1000_month"], color="grey", ls=":", lw=1.2, label="age-based replacement")
        a.set_xscale("log"); a.set_xlabel("alert threshold on calibrated 30-day failure probability"); a.set_title("remote confirmation " + ("ON" if conf else "OFF")); a.grid(alpha=0.3)
    ax[0].set_ylabel("expected cost per 1,000 systems per month (HYPOTHETICAL units)"); ax[0].legend(fontsize=8)
    fig.suptitle("Cost vs alert threshold, test cohort, unlimited technician capacity. HYPOTHETICAL costs: shapes only")
    fig.tight_layout(); fig.savefig(FIG / "cost_vs_threshold.png", dpi=150); plt.close(fig)
    h = out[out.kind == "model_selected_on_calibration"]
    h = h[(h.confirm == False) & (h.K == -1)].sort_values("EC_per_1000_month").iloc[0]
    bidx, bucket, score, tp, lead_, flips = te.events(h.tau, int(h.N), c0["cooldown_days"])
    fig, a = plt.subplots(figsize=(6.5, 4)); a.hist(lead_[tp], bins=np.arange(0, W + 7, 7), color="#1f5fa8")
    a.set_xlabel("days between alert and repair (true alerts)"); a.set_ylabel("alerts"); a.set_title(f"Lead time, {h.policy}\n(test cohort; hypothetical-cost-selected operating point)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "leadtime_hist.png", dpi=150); plt.close(fig)
    fig, a = plt.subplots(figsize=(5.5, 5)); s_te = sc[sc.ID.isin(sp["test"]) & sc.valid30]
    for col in ("raw", "platt", "iso"):
        q = pd.qcut(s_te[col], 10, duplicates="drop"); gg = s_te.groupby(q, observed=True).agg(p=(col, "mean"), y=("y30", "mean")); a.plot(gg.p, gg.y, "o-", label=col)
    mx = float(s_te[["raw", "platt", "iso"]].max().max()); a.plot([0, mx], [0, mx], "k:", lw=1); a.set_xlabel("predicted 30-day failure probability"); a.set_ylabel("observed frequency"); a.legend(); a.set_title("Reliability, test cohort")
    fig.tight_layout(); fig.savefig(FIG / "calibration_bboxx.png", dpi=150); plt.close(fig)
    fig, a = plt.subplots(figsize=(6, 4.6)); im = a.imshow(sav, origin="lower", cmap="RdBu", vmin=-np.abs(sav).max(), vmax=np.abs(sav).max(), aspect="auto")
    a.set_xticks(range(len(vs)), [f"x{v}" for v in vs]); a.set_yticks(range(len(bs)), [f"x{b}" for b in bs]); a.set_xlabel("technician trip cost multiplier"); a.set_ylabel("failure base rate multiplier (1 = as sampled)")
    a.contour(sav, levels=[0], colors="k"); fig.colorbar(im, label="saving vs reactive (hypothetical units/1,000/month)"); a.set_title("Break-even region (black line = 0; blue = prediction pays)")
    fig.tight_layout(); fig.savefig(FIG / "breakeven_bboxx.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
