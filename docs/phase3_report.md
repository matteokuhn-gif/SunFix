# Phase 3 report: EOL prediction and benchmark (all 1,027 batteries)

Run: `python -m sunfix.phase3_features; python -m sunfix.classify; python -m sunfix.phase3_figure; python -m sunfix.make_pdf`.
Outputs: `results/phase3_results.csv` (all sets, models, horizons), `results/phase3_predictions.parquet`, `results/figures/phase3_summary.png`, `results/SunFix_findings_summary.pdf`.

## Protocol (Aitio's, as far as stated)
5-fold stratified CV by battery (seed 0). Per fold the resistance model hyperparameters are fitted on 8 training-fold batteries (label-free) and applied to all batteries. Decision time = end of series minus h days (failed: repair date; alive: last telemetry); features use only data up to that time. Pooled out-of-fold predictions over all 1,027 batteries; threshold 0.5. Aitio's extra test subsets (40/60/80% of failures) leave expected balanced accuracy unchanged and are not reproduced.

## Headline (GPC, resistance without availability columns + usage factors)
Balanced accuracy 81.3 / 79.1 / 78.6 / 76.2 / 74.2 % at h = 0 / 14 / 28 / 42 / 56 days (paper 82 / 80 / 76 / 76 / 73). 95% bootstrap interval about +-2.5 points; fold-to-fold sd 1.7-2.9 points. Logistic regression and gradient boosting agree within 1-2 points.
Usage only 71.6 / 70.5 / 70.7 / 70.5 / 70.5; resistance only 73.8 / 70.4 / 69.9 / 67.6 / 64.3; age alone 61.8.
Calibration is good (ECE 0.02-0.03, Brier 0.13-0.17). Sensitivity 79-72%, specificity 84-77%. False alarms: 162-235 healthy batteries flagged per 1,000 per decision (at threshold 0.5). Precision 79-74% at the test prevalence of 48% failed; at 5% failures precision falls to about 17% (28-day model).

## Leak found and removed
The first resistance feature set included `unc` and `stale_days` (days since the last charging session). `stale_days` alone gives 69% at h = 0 and chance at the other horizons, because for failed batteries the series end (repair date) can be later than the last telemetry, while for alive batteries the end is the last telemetry by construction. With these columns the combined model scored 87.0% at h = 0 (the other horizons barely changed). They are excluded from the headline feature set (`res_core`). Aitio's protocol has the same structural issue; their requirement that the last qualifying segment is within 30 days of the end limits it.

## Caveats
- Hyperparameters of the resistance GP vary between folds (sf0 0.6-6.4), because 8 training batteries determine them poorly; a larger fitting set is cheap.
- Hyperparameters are fitted on training-fold batteries using their whole series, including data after the training batteries' own decision times. This is label-free and does not touch test batteries.
- Labels are customer repair decisions.
- Not done: Aitio's random-walk benchmark model; Phase 4 minimum-data experiments.
