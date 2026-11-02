import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import ROOT

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 200})
H = [0, 14, 28, 42, 56]
AITIO = {"res+stress": [82, 80, 76, 76, 73], "res": [76, 73, 67, 64, 63], "stress": [71, 70, 71, 69, 66]}


def main():
    r = pd.read_csv(ROOT / "results" / "phase3_results.csv")
    g = lambda s, m="gpc": r[(r.set == s) & (r.model == m)].sort_values("h")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    ours = g("res_core+stress")
    a.fill_between(H, ours.ba_lo * 100, ours.ba_hi * 100, color=BLUE, alpha=.15, lw=0)
    a.plot(H, ours.balanced_acc * 100, color=BLUE, lw=2, marker="o", ms=4)
    a.plot(H, AITIO["res+stress"], color=ORANGE, lw=2, marker="s", ms=4)
    a.plot(H, g("stress").balanced_acc * 100, color=AQUA, lw=1.6, marker="o", ms=3)
    a.plot(H, g("res_core").balanced_acc * 100, color=MUTED, lw=1.4, ls="--", marker="o", ms=3)
    a.axhline(70, color=INK, lw=.8, ls=":"); a.text(1, 69.6, "70% 'useful' line", ha="left", va="top", color=INK, fontsize=7)
    a.text(1, 83.5, "Aitio & Howey (paper)", color=ORANGE, fontsize=8, va="bottom")
    a.text(1, 75.2, "this work: resistance + usage", color=BLUE, fontsize=8, va="top")
    a.text(57, 62.5, "resistance only", color=MUTED, fontsize=8, ha="right", va="top")
    a.text(57, 71.2, "usage only", color=AQUA, fontsize=8, ha="right", va="bottom")
    a.set_xlabel("warning time: days before the end of the series"); a.set_ylabel("balanced accuracy (%)")
    a.set_xlim(-2, 58); a.set_ylim(58, 90); a.set_title("Prediction accuracy vs warning time", loc="left", color=INK, fontsize=9)
    p = pd.read_parquet(ROOT / "results" / "phase3_predictions.parquet")
    b = ax[1]
    for h, ls, c in ((28, "-", BLUE), (56, "--", BLUE)):
        d = p[(p.h == h) & (p.set == "res_core+stress") & (p.model == "gpc")]
        bins = np.quantile(d.p, np.linspace(0, 1, 9)); bins[-1] += 1e-9
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = d[(d.p >= lo) & (d.p < hi)]; xs.append(s.p.mean()); ys.append(s.failed.mean())
        b.plot(xs, ys, color=c, ls=ls, marker="o", ms=3, lw=1.6)
        b.text(xs[-1] - .02, ys[-1] - .06 if h == 28 else ys[-1] + .02, f"{h} d", color=INK, fontsize=7, ha="right")
    b.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
    b.set_xlabel("predicted probability of failure"); b.set_ylabel("observed share failed"); b.set_title("Calibration", loc="left", color=INK, fontsize=9)
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "phase3_summary.png"); print("ok")


if __name__ == "__main__":
    main()
