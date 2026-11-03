# Phase 6b report: benchmark models on the same landmarks

Run: `python -m sunfix.features6; python -m sunfix.bench; python -m sunfix.bench_ibs` (about 25 min). Outputs: `results/benchmark_table.csv`, `shap_lightgbm_bboxx.csv`, `benchmark_ibs.csv`, `figures/shap_lightgbm_bboxx.png`. Operating-cost columns (`ops_*`) use the **hypothetical** costs of `config/costs.yaml`; they are shapes, not findings.

## Set-up
Same landmarks (434k system-days, age >= 120 d), same temporal split (train / calibration / test by activation date) and same ops evaluation as 6a. Features (`sunfix/features6.py`, no look-ahead; day d uses only complete days before d and resistance-trajectory points up to d): GP resistance level, slopes (7/30/90 d plus Phase 3's), curvature, CUSUM statistic, fleet z-score; equivalent full cycles, partial-charge Ah in the 12-13 V band, end-of-discharge V at 0.5-1.5 A, hours at low V while discharging, low-voltage trips, hours above 35 / 40 degC. Availability features (coverage, outage days, gap hours, resistance variance) are an ablation group only, because availability leaked in Phase 3.
Models: GP classifier (2,000 sampled landmarks; exact GPC cannot take 200k rows); LightGBM per horizon (30, 56 d); Cox PH and random survival forest on landmarks every 14 days with censoring of alive batteries; unsupervised CUSUM + fleet z-score (ranks from training cohort, no labels); 2-of-3 ensemble (median of calibrated LightGBM, RSF, CUSUM+z). All scores Platt-calibrated on the calibration cohort; balanced-accuracy threshold chosen on calibration.
**Not implemented:** Bayesian online change-point detection (CUSUM only); scikit-survival's own cross-validated tuning; fleet z-score uses only peers installed in the 60 days before the battery (causal), not the brief's +-1 month window.

## Result: use the at-risk column
Balanced accuracy over all landmarks is 88-91% for every supervised model. **That is inflated**: no battery fails before 360 days of age, so predicting "healthy" for young systems is trivially right (SHAP: age 2.33 log-odds, next feature 0.30). The informative columns are restricted to age >= 360 d (the at-risk period), thresholds re-chosen on the calibration cohort:
| Model (30-day label) | Balanced acc., at-risk | AUC, at-risk | C-index | Brier (all) |
|---|---|---|---|---|
| LightGBM (resistance + usage) | 77.6% | 0.864 | 0.837 | 0.020 |
| LightGBM usage-only | 76.2% | 0.857 | 0.840 | 0.020 |
| LightGBM age only | 64.9% | 0.687 | 0.696 | 0.025 |
| LightGBM + availability (leak check) | 78.5% | 0.871 | 0.845 | 0.019 |
| GP classifier (sampled) | 75.2% | 0.852 | 0.748 | 0.021 |
| Cox PH | 76.9% | 0.864 | 0.881 | 0.024 |
| Random survival forest | 76.6% | 0.851 | 0.874 | 0.021 |
| CUSUM + fleet z (unsupervised) | 58.0% | 0.570 | 0.499 | 0.028 |
| 2-of-3 ensemble | 75.9% | 0.847 | 0.865 | 0.020 |
At 56 days the at-risk balanced accuracy is 74-76% for the supervised models (age only: 53%). Integrated Brier score 7-56 d: Cox 0.0254, RSF 0.0223 (lower is better). Time-dependent AUC at the horizon is 0.95-0.97 but inflated by age, so use the at-risk AUC. Roughly 100 test failures, so differences of about 2 points or less between supervised models are not distinguishable.

## What it means
1. **The supervised models are interchangeable** (75-78% at-risk). The simplest (logistic-style/Cox or LightGBM) does as well as the GP classifier and the survival forest. This agrees with Phase 3 (78.6% at 28 d on a 50/50 sample).
2. **Usage statistics carry most of the signal beyond age**: age only 65%, usage-only 76%, plus resistance 78%. The resistance features add about 1.4 points at 30 days, within noise. This is the same finding as Phase 3's usage-only baseline of about 70.5%, in a more demanding test.
3. **The unsupervised CUSUM/fleet-z detector is not useful alone** (58%): single-battery resistance trajectories are noisy.
4. **The 2-of-3 ensemble does not beat its best member.**
5. **No sign of an availability leak in landmark scoring** (+0.9 points with availability features), but not proven absent.
6. SHAP (LightGBM, 30 d): age 2.33, end-of-discharge V 0.30, cycles 0.30, partial-charge Ah 0.13, resistance level 0.12, fleet z 0.11.

## Operating metrics at each model's cost-optimal point (hypothetical costs; 30-day scores; test cohort; per 1,000 systems per month)
Without confirmation the models sit at 0.3-5 false dispatches with 19-25 missed failures (of 29.5), medians of 8-23 days warning. With remote confirmation: 0.8-2.7 false, 12-15 missed, median 22-32 days warning. Expected cost 3,437-3,490 for the supervised models with confirmation versus 3,832 reactive. The differences between models are well inside the bootstrap uncertainty found in 6a (several hundred cost units), so **no model can be called cheapest**. The age-only model gets essentially no saving (3,793), the unsupervised model little (3,736).

## Limitations
Hypothetical costs; the test cohort is younger than the training cohort; correlated landmarks (many rows per battery) make ranking metrics optimistic and the survival-model standard errors invalid; only about 100 failed test batteries; GPC trained on a 2,000-landmark sample; one resistance-GP fit per fold from Phase 3.
