# SunFix: predicting solar-home-system battery failure from telemetry (Phases 0-5)

Research code and write-up. Data: Aitio & Howey (2021), Bboxx lead-acid batteries, CC BY-NC 4.0 (research use only; the paper lists a patent application, GB 2105995.1). Repo: github.com/matteokuhn-gif/SunFix (private). Details: `docs/`, `METHODOLOGY.md`, `LOG.md`.

## 1. What was done
We applied BattGP's Gaussian-process resistance model (Schaeffer et al. 2024) to the 1,027 lead-acid batteries of the Bboxx field dataset (491 entered repair, 536 censored alive; about 620M telemetry rows). For each battery we estimate internal resistance over time from charging sessions, build features available at a decision time, and predict "this battery fails within h days". Evaluation follows Aitio & Howey: 5-fold cross-validation by battery, balanced accuracy, hyperparameters fitted inside each training fold. We then thinned the telemetry in 15 ways to find the minimum data needed.

In plain terms: the model watches how much voltage a battery loses per amp of charging current. Failing batteries show a rising "resistance" in the last weeks. Combined with how hard the battery has been used, this predicts failure better than usage alone.

## 2. Reproduction against the paper (all 1,027 batteries)
| Days before failure | This work | Aitio & Howey |
|---|---|---|
| 0 | 81.3% | 82% |
| 28 | 78.6% | 76% |
| 56 | 74.2% | 73% |
Balanced accuracy, GP classifier on resistance plus usage features; 95% interval about +-2.5 points. We reproduce the paper within noise. Calibration is good (ECE 0.02-0.03).
Deviations (full list in METHODOLOGY.md): BattGP is not OCV-free in practice, so we rebuilt state of charge by Coulomb counting with the lab OCV curve; we added a per-segment offset to the resistance model; failed batteries are truncated at the repair date (their telemetry continues a median 89 days afterwards); a data-availability leak (`stale_days`, `unc`) was found and removed (it inflated the score at failure from 81% to 87%).
**Honest baseline:** usage statistics alone (throughput, temperature, cycles, age) already give about 70.5%; resistance adds roughly 5-11 points. The 70% "useful" bar is therefore weak evidence by itself.

## 3. Minimum-data results (Phase 4; 40% subsample, about +-4 points)
Four-week warning, balanced accuracy of the full model (usage-only in brackets):
| Telemetry | 28 days |
|---|---|
| Native, 1 reading/min | 78% (71) |
| 1 reading per 5 / 15 / 60 min | 79 / 79 / 78% (70 / 70 / 69) |
| One daily record (min/max/mean V, I, T, Wh in/out) | 79% (70); 75% on all batteries |
| No temperature | 81% (71) |
| History 30 / 90 / 180 days | 74 / 78 / 78% (61 / 63 / 66) |
| Voltage rounded to 0.1 V, current to 0.1 A | 77 / 78% (72 / 71) |
| 10 / 30 / 50% of rows lost | 77 / 78 / 77% (69 / 69 / 68) |

**Smallest specification tested that still gives a useful 30-day warning: one daily summary record per system**, with a gain over usage-only of about 9 points at 28 days. No tested degradation produced a loss larger than the +-4 point uncertainty. At 56 days the benefit over usage statistics is not established. Combinations of degradations were not tested. False alarms are the open issue: about 180-250 of 1,000 healthy batteries are flagged per decision at threshold 0.5 (measured on a 50/50 failed/healthy sample), so at realistic failure rates most alerts would be wrong (Phase 6).

## 4. Limitations
- Lead-acid, not LFP, which is what new solar home systems ship with. One manufacturer, one data set.
- Alive batteries are censored; labels are customer repair decisions, not confirmed battery faults.
- Evaluation at about 50% failure prevalence overstates precision in service.
- Resistance GP hyperparameters are fitted on 8 batteries per fold and vary between folds; Phase 4 reused them for degraded data. Phase 4 ran on a 40% subsample.
- Daily-summary scenario uses a crude resistance proxy; degradations are emulated from 1-minute data (random, not bursty, loss).
- Licence: CC BY-NC data and a patent application mean this is research code only.

## 5. Next steps
1. Cost-aware evaluation at realistic failure rates and benchmark models (Phase 6a-6c).
2. LFP validation on the BattGP field data (Zenodo 13715694), including a pack-only test of whether cheap pack-level telemetry detects cell faults (6d), then RWTH home-storage data (6e).
3. Operator data: repair records with a battery/other/no-fault outcome, trip costs, and a 2-3 month shadow-mode trial before any dispatch (6g).
