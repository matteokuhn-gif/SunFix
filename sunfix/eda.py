"""Phase 1 EDA: figures in results/figures/ and numbers for docs/eda_report.md."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clean import clean_raw, segments_for
from .load import ROOT, cached_raw, dev_subset, load_meta

FIG = ROOT / "results" / "figures"
ALIVE, FAILED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130})


def legend(ax):
    ax.legend(handles=[plt.Line2D([], [], color=ALIVE, lw=6, label="alive (censored)"),
                       plt.Line2D([], [], color=FAILED, lw=6, label="failed (repair)")], frameon=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m, s = load_meta(), pd.read_csv(ROOT / "data" / "summary_all.csv", index_col=0)
    x = m.join(s)
    ids = dev_subset()
    stats = {}

    # sampling interval, V/I/T distributions from the dev subset (raw, before cleaning)
    dts, V, I, T = [], [], [], []
    for i in ids:
        d = cached_raw(i)
        dts.append(np.diff(d.t.to_numpy())); V.append(d.V.to_numpy()[::20]); I.append(d.I.to_numpy()[::20]); T.append(d["T"].to_numpy()[::20])
    dt = np.concatenate(dts); V, I, T = map(np.concatenate, (V, I, T))
    stats["dt_share_60s"] = float((dt == 60).mean()); stats["dt_share_le120s"] = float((dt <= 120).mean())
    stats["dt_share_gt610s"] = float((dt > 610).mean()); stats["dt_share_gt1day"] = float((dt > 86400).mean())
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].hist(np.log10(dt[dt > 0]), bins=80, color=ALIVE); ax[0].set_yscale("log"); ax[0].set_xlabel("log10 sampling interval (s)")
    ax[0].set_title("Sampling interval (mode 60 s)", loc="left", color=INK)
    for a, v, lab, t in zip(ax[1:], (V, I, T), ("V", "A (negative = charging)", "degC"), ("Voltage", "Current", "Temperature")):
        a.hist(v, bins=100, color=ALIVE); a.set_yscale("log"); a.set_xlabel(lab); a.set_title(t, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "eda_distributions.png"); plt.close(fig)

    # rows, lifetime, gaps over all 1027
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for flag, c in ((False, ALIVE), (True, FAILED)):
        g = x[x.failed == flag]
        ax[0].hist(g.rows / 1e3, bins=30, alpha=0.7, color=c); ax[1].hist(g.Lifetime, bins=30, alpha=0.7, color=c)
        ax[2].hist(np.log10(g.gap_max_d), bins=30, alpha=0.7, color=c)
    ax[0].set_xlabel("raw rows per battery (thousand)"); ax[1].set_xlabel("lifetime (days, activation to repair / censoring)")
    ax[2].set_xlabel("log10 longest telemetry gap (days)")
    for a, t in zip(ax, ("Rows per battery", "Lifetime", "Longest gap")): a.set_title(t, loc="left", color=INK)
    legend(ax[1]); fig.tight_layout(); fig.savefig(FIG / "eda_rows_lifetime_gaps.png"); plt.close(fig)

    # telemetry vs repair date
    x["post_repair_days"] = (x.t1 - x.failure_time.map(lambda v: v.timestamp() if pd.notna(v) else np.nan)) / 86400
    f = x[x.failed]
    fig, ax = plt.subplots(figsize=(6, 3.4)); ax.hist(f.post_repair_days, bins=40, color=FAILED)
    ax.axvline(0, color=INK, lw=1); ax.set_xlabel("days telemetry continues after repair date"); ax.set_title("Failed batteries: data after repair", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "eda_post_repair.png"); plt.close(fig)
    stats["post_repair_median_d"] = float(f.post_repair_days.median()); stats["post_repair_gt0_share"] = float((f.post_repair_days > 0).mean())

    # five example series each: daily mean voltage and charging current vs days to end (repair or end of data)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.2), sharex="col")
    for col, flag in enumerate((True, False)):
        pick = rng.choice([i for i in ids if m.failed[i] == flag], 5, replace=False)
        for i in pick:
            d = clean_raw(i, m); end = d.t.iloc[-1]
            day = ((d.t - end) // 86400).astype(int)
            g = d.groupby(day).agg(V=("V", "mean"), I=("I", lambda v: v[v < -0.05].mean()))
            c = FAILED if flag else ALIVE
            ax[0, col].plot(g.index, g.V, color=c, lw=0.9, alpha=0.85); ax[1, col].plot(g.index, g.I, color=c, lw=0.9, alpha=0.85)
        ax[0, col].set_title(("Failed" if flag else "Alive") + ": 5 examples, aligned to " + ("repair date" if flag else "end of data"), loc="left", color=INK)
        ax[1, col].set_xlabel("days before end")
    ax[0, 0].set_ylabel("daily mean voltage (V)"); ax[1, 0].set_ylabel("daily mean charging current (A)")
    fig.tight_layout(); fig.savefig(FIG / "eda_examples.png"); plt.close(fig)

    # cleaning / segment tally on the dev subset
    rows = []
    for i in ids:
        raw = cached_raw(i); c = clean_raw(i, m); sg = segments_for(i, m)
        rows.append(dict(ID=i, failed=m.failed[i], raw=len(raw), after_clean=len(c), seg_rows=len(sg), n_seg=sg.seg.nunique()))
    r = pd.DataFrame(rows); r.to_csv(ROOT / "data" / "dev_cleaning_tally.csv", index=False)
    stats["dev_tally"] = r.groupby("failed")[["raw", "after_clean", "seg_rows", "n_seg"]].mean().round(0).to_dict()
    stats["dev_dropped_nonservice_share"] = float(1 - r.after_clean.sum() / r.raw.sum())
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for flag, c in ((False, ALIVE), (True, FAILED)): ax.hist(r[r.failed == flag].n_seg, bins=15, alpha=0.7, color=c)
    ax.set_xlabel("qualifying charging segments per battery (dev subset)"); ax.set_title("Segments per battery (Aitio mean: 261)", loc="left", color=INK); legend(ax)
    fig.tight_layout(); fig.savefig(FIG / "eda_segments.png"); plt.close(fig)

    stats["rows_all_total"] = int(x.rows.sum()); stats["start_before_activation_n"] = int((x.t0 < x.activation_time.map(lambda v: v.timestamp()) - 86400).sum())
    stats["gaps_gt30d_batteries"] = int((x.gaps_gt30d > 0).sum()); stats["V_lt9_or_gt16_rows_dev"] = int(((V < 9) | (V > 16)).sum())
    stats["lifetime"] = x.groupby("failed").Lifetime.describe().round(0).to_dict()
    (ROOT / "data" / "eda_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
