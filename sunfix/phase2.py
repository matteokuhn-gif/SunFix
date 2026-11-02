"""Phase 2: run the resistance GP on the 60 development batteries for both variants, check sanity, write figures."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from .load import ROOT, dev_subset, load_meta
from .resistance import Normaliser, prepare, run_battery

RES = ROOT / "data" / "resistance"
FIG = ROOT / "results" / "figures"
ALIVE, FAILED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130})


def load_hyper(variant):
    h = json.loads((ROOT / "data" / f"hyper_{variant}.json").read_text())
    return h, Normaliser(np.array(h["norm_mean"]), np.array(h["norm_std"]), np.array(h["basis"]))


def one(i, variant, h, norm, m):
    t0 = time.time()
    p = prepare(i, variant, m)
    t1 = time.time()
    out, ll, sl = run_battery(p, norm, h["sf0"], h["sf1"], tuple(h["ls"]), slices=True, tau=h.get("tau", 0.0))
    t2 = time.time()
    out["ID"] = i
    d = RES / variant; d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"{i}.parquet", index=False)
    np.savez(d / f"{i}_slices.npz", **{f"{k}_{f}": v for k, s in sl.items() for f, v in s.items()})
    return dict(ID=i, rows=len(p.y), segs=len(out), ll=ll, ll_per_row=ll / len(p.y), t_prep=t1 - t0, t_gp=t2 - t1,
                v_rmse=float(np.sqrt(np.average(out.innov_v_rmse**2, weights=out.n))), innov_z=float(np.average(out.innov_z, weights=out.n)))


def separation(variant, m, ids):
    rows = []
    for i in ids:
        r = pd.read_parquet(RES / variant / f"{i}.parquet")
        t = r.t * 400.0   # days since activation
        base = r.r0[(t >= 60) & (t <= 120)].median()
        end = r.r0[t >= t.iloc[-1] - 30].median()
        rows.append(dict(ID=i, failed=bool(m.failed[i]), r_end=end, r_base=base, r_ratio=end / base, r_rise=end - base,
                         slope_end=r.z1[t >= t.iloc[-1] - 60].median() / 400.0 * 1000,   # mOhm/day
                         sd_end=np.sqrt(r.r0var[t >= t.iloc[-1] - 30].median()), r_early_sd=r.r0.diff().abs().median()))
    return pd.DataFrame(rows).set_index("ID")


def trajectory_figure(variant, m, ids, rng):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    curves = {True: [], False: []}
    for flag, a, c in ((True, ax[0], FAILED), (False, ax[1], ALIVE)):
        pick = rng.choice([i for i in ids if m.failed[i] == flag], 10, replace=False)
        for i in pick:
            r = pd.read_parquet(RES / variant / f"{i}.parquet")
            a.plot(r.t * 400 - r.t.iloc[-1] * 400, r.r0, color=c, lw=0.9, alpha=0.8)
        a.set_title(("Failed: aligned to repair date" if flag else "Alive: aligned to end of data") + ", 10 random batteries", loc="left", color=INK)
        a.set_xlabel("days before end"); a.set_ylim(0, 1.6)
    ax[0].set_ylabel("R0 at reference operating point (Ohm)")
    grid = np.arange(-400, 1, 10)
    for flag, c, lab in ((True, FAILED, "failed"), (False, ALIVE, "alive")):
        M = []
        for i in [j for j in ids if m.failed[j] == flag]:
            r = pd.read_parquet(RES / variant / f"{i}.parquet"); x = r.t * 400 - r.t.iloc[-1] * 400
            base = r.r0[(r.t * 400 >= 60) & (r.t * 400 <= 120)].median()
            M.append(np.interp(grid, x, r.r0 / base, left=np.nan))
        M = np.array(M)
        ax[2].plot(grid, np.nanmedian(M, 0), color=c, lw=2); ax[2].fill_between(grid, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=c, alpha=0.18, lw=0)
        ax[2].text(grid[-1] - 5, np.nanmedian(M, 0)[-1], lab, color=INK, ha="right", va="bottom")
    ax[2].set_title("Median and IQR, relative to own early-life level", loc="left", color=INK); ax[2].set_xlabel("days before end"); ax[2].set_ylabel("R0 / R0 at days 60-120")
    ax[2].set_ylim(0.4, 2.2)
    fig.suptitle(f"Variant '{variant}'", x=0.01, ha="left", color=INK); fig.tight_layout(); fig.savefig(FIG / f"phase2_trajectories_{variant}.png"); plt.close(fig)


def main():
    m = load_meta(); ids = dev_subset(); FIG.mkdir(parents=True, exist_ok=True)
    summary = {}; seps = {}
    for variant in ("ocv", "nocv"):
        h, norm = load_hyper(variant)
        t = time.time()
        stats = pd.DataFrame(Parallel(n_jobs=6)(delayed(one)(i, variant, h, norm, m) for i in ids)).set_index("ID")
        wall = time.time() - t
        # clean single-process timing on 6 batteries (already cached)
        t = time.time(); [one(i, variant, h, norm, m) for i in ids[:3] + ids[-3:]]; seq = (time.time() - t) / 6
        sep = separation(variant, m, ids); seps[variant] = sep
        res = {"seconds_per_battery_sequential": seq, "projected_hours_all_1027_sequential": seq * 1027 / 3600,
               "wall_s_60_batteries_6_procs": wall, "v_rmse_mean": float(stats.v_rmse.mean()), "innov_z_mean": float(stats.innov_z.mean()),
               "ll_per_row_mean": float(stats.ll_per_row.mean())}
        for col in ("r_end", "r_ratio", "r_rise", "slope_end"):
            res[f"auc_{col}"] = float(roc_auc_score(sep.failed, sep[col]))
            res[f"median_{col}_failed"] = float(sep[sep.failed][col].median()); res[f"median_{col}_alive"] = float(sep[~sep.failed][col].median())
        summary[variant] = res
        stats.to_csv(ROOT / "data" / f"phase2_stats_{variant}.csv"); sep.to_csv(ROOT / "data" / f"phase2_sep_{variant}.csv")
        trajectory_figure(variant, m, ids, np.random.default_rng(1))
        print(variant, json.dumps(res, indent=1), flush=True)
    (ROOT / "data" / "phase2_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
