"""Phase 4: assemble results/minimum_data_table.csv and results/figures/phase4_minimum_data.png from data/phase4/*.csv."""
import numpy as np
import pandas as pd

from .phase4 import FRAC, HS, OUT, SCENARIOS, native_row
from .load import ROOT

USEFUL = 0.70          # brief: balanced accuracy >= 70%
HEADLINE_H = 28        # "30-day warning" = decision 28 days before the end of the series
FULL = {"res_core+stress", "res_core+stress_noT", "proxy+stress"}   # model using everything the scenario allows
USAGE = {"stress_only", "stress_only_noT", "stress_only_daily"}


def build():
    parts = ([native_row()] if FRAC >= 1 else []) + [pd.read_csv(OUT / f"{n}.csv") for n in SCENARIOS if (OUT / f"{n}.csv").exists()]
    r = pd.concat(parts, ignore_index=True)
    r["role"] = np.where(r.set.isin(FULL), "full", np.where(r.set.isin(USAGE), "usage_only", np.where(r.set == "age_only", "age_only", "resistance_only")))
    cols = ["scenario", "group", "h", "set", "role", "model", "balanced_acc", "ba_lo", "ba_hi", "sensitivity", "specificity", "precision",
            "false_alarms_per_1000_per_month", "coverage", "samples_per_day"]
    r = r[cols].sort_values(["h", "group", "scenario", "role", "model"])
    r.to_csv(ROOT / "results" / f"minimum_data_table{'' if FRAC >= 1 else '_sub' + str(int(FRAC * 100))}.csv", index=False)
    # headline view: GPC full model vs usage-only (LR for usage-only, as in Phase 3 headline comparisons use best of both)
    g = r[(r.h == HEADLINE_H) & (r.model == "gpc") & (r.role == "full")].set_index("scenario")
    u = r[(r.h == HEADLINE_H) & (r.role == "usage_only") & (r.model == "lr")].set_index("scenario")
    head = g[["group", "balanced_acc", "ba_lo", "ba_hi", "sensitivity", "specificity", "false_alarms_per_1000_per_month", "coverage", "samples_per_day"]].join(
        u[["balanced_acc"]].rename(columns={"balanced_acc": "usage_only_ba"}))
    head["gain_over_usage"] = head.balanced_acc - head.usage_only_ba
    head["useful"] = head.balanced_acc >= USEFUL
    head.sort_values("samples_per_day", ascending=False).to_csv(ROOT / "results" / f"minimum_data_headline_28d{'' if FRAC >= 1 else '_sub' + str(int(FRAC * 100))}.csv")
    print(head.sort_values("samples_per_day", ascending=False).round(3).to_string())
    figure(r)


def figure(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    order = [n for n in ["native_1min"] + list(SCENARIOS) if n in set(r.scenario)]
    order = list(dict.fromkeys(order))
    for ax, h in zip(axes, HS):
        for j, n in enumerate(order[::-1]):
            a = r[(r.scenario == n) & (r.h == h) & (r.model == "gpc") & (r.role == "full")]
            u = r[(r.scenario == n) & (r.h == h) & (r.model == "lr") & (r.role == "usage_only")]
            if len(a):
                ax.errorbar(a.balanced_acc.iloc[0], j, xerr=[[a.balanced_acc.iloc[0] - a.ba_lo.iloc[0]], [a.ba_hi.iloc[0] - a.balanced_acc.iloc[0]]],
                            fmt="o", color="#1f5fa8", capsize=3, label="all data the scenario allows" if j == 0 else None)
            if len(u):
                ax.plot(u.balanced_acc.iloc[0], j, "s", color="#999999", label="usage statistics only" if j == 0 else None)
        ax.axvline(USEFUL, color="#c0392b", ls="--", lw=1, label="useful (70%)")
        ax.set_yticks(range(len(order)), order[::-1])
        ax.set_title(f"{h} days before failure")
        ax.set_xlabel("balanced accuracy")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("SunFix Phase 4: prediction quality as telemetry is thinned (95% bootstrap interval)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / f"phase4_minimum_data{'' if FRAC >= 1 else '_sub' + str(int(FRAC * 100))}.png", dpi=160)


if __name__ == "__main__":
    build()
