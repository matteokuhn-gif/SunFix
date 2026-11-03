# Phase 6a report: cost-aware evaluation (Bboxx lead-acid)

**ALL COSTS ARE HYPOTHETICAL** (`config/costs.yaml`, to be replaced with operator data). Cost-based numbers below show mechanics and shapes, not findings. The non-cost numbers (hazard, calibration, alerts, false dispatches, lead times) do not depend on the placeholder costs, except where the operating point was chosen by cost.
Run: `python -m sunfix.landmarks; python -m sunfix.run_6a` (about 4 min). Code: `sunfix/landmarks.py`, `sunfix/eval_ops.py`, `sunfix/run_6a.py`.
Outputs: `results/ops_metrics_bboxx.csv`, `ops_sensitivity_bboxx.csv`, `ops_breakeven_bboxx.csv`, `calibration_bboxx.csv`, `hazard_by_age_bboxx.csv`; figures `cost_vs_threshold.png`, `leadtime_hist.png`, `calibration_bboxx.png`, `breakeven_bboxx.png`.

## Set-up
- 434,045 landmarks = every system-day at age >= 120 days for all 1,027 batteries. Label "fails within 30 days"; alive batteries' last 30 days are dropped (outcome unknown). Features as of that day: resistance level/slopes/relative level (out-of-fold causal trajectories from Phase 3) plus usage statistics; no look-ahead.
- Split by battery by activation date: earliest 50% train (513 batteries, 60% failed), next 20% calibration (205, 43%), latest 30% test (309, 30%). The falling failed share is censoring: later batteries are younger when the data ends. Consequence: the test cohort is observed only at younger ages. Leave-one-district-out is **not possible** (no location metadata).
- Model: gradient boosting; Platt calibration on the calibration cohort. Test: Brier 0.021, ECE 0.006; observed 30-day rate 2.9% vs mean prediction 2.3% (slight under-prediction in the later cohort).
- Policy: alert when calibrated score >= tau for N consecutive days; optional remote confirmation; capacity K dispatches per 1,000 systems per month; 90-day cooldown; an alert is true if repair follows within 56 days. Expected-value accounting (no random draws). Operating point chosen on the calibration cohort, reported on the test cohort with a battery-level bootstrap (150 draws).

## Empirical hazard (as sampled)
Zero failures below about 360 days of age (all lifetimes are >= 401 days). Monthly failure hazard by age: 4.9% (360-419 d), 9.1%, 11.6%, 12.5%, 13.7%, 14.1% (660+). Overall 3.6% per system-month in the landmark set, about 3.0% in the test cohort. **The sample is about 48% failed batteries by design, so this overstates a real fleet.** Base-rate sensitivity handles this (failed batteries re-weighted).

## Lead with the operational numbers (test cohort, as-sampled base rate, per 1,000 systems per month)
| Policy | False dispatches | True dispatches | Missed failures | Precision | Median warning |
|---|---|---|---|---|---|
| Phase 3 style (threshold maximising balanced accuracy, N=1) | **67** | 17 | 12 | 21% | 33 d |
| Elkan cost threshold (p* = 0.57), N=1 | 2 | 7 | 23 | 75% | 23 d |
| Cost-optimal on calibration (tau 0.11, N=7, remote confirmation on) | 2 | 16 | 14 | 88% | 28 d |
| Reactive only (do nothing) | 0 | 0 | 29 | n/a | n/a |
Plain reading: re-expressed at realistic prevalence, Phase 3's balanced operating point sends about **4 wasted trips for every useful one** (67 vs 17), at the as-sampled failure rate. At a real fleet's much lower failure rate the ratio is worse (precision falls roughly in proportion to the base rate). Raising the threshold removes most false trips but also most of the catches (Elkan point: 7 of about 29 failures caught).
Bootstrap interval for one policy (N=3, no confirmation, unlimited capacity): expected cost 3,766 (3,197-4,355), missed failures 21-29 per 1,000 per month. Intervals are wide: 93 failed test batteries.

## Cost-based comparison (hypothetical units per 1,000 systems per month)
Reactive 3,832; age-based replacement 3,832 (best age chosen on the training cohort is 740 d, i.e. practically never replace; with these placeholders replacing early saves only 30 per failure, because planned (100) and emergency (130) costs are close); Elkan 3,722; best cost-optimal policy with confirmation 3,479-3,490 (about 9% below reactive); Phase 3 style 5,998 (**56% above reactive**: acting on balanced-accuracy alerts costs more than doing nothing).
Shape of the curves (`cost_vs_threshold.png`): without confirmation, alerting at low thresholds costs more than reactive; persistence N helps a little at low thresholds; remote confirmation flattens the curve and creates a small saving across a wide threshold range. A technician-capacity cap (K = 5-20) pushes the optimum to high thresholds because failures arrive in calendar bursts.

## Sensitivity (+-50% each cost, and base rate; operating point re-selected on calibration each time)
- The saving from the best policy versus reactive is positive for almost every perturbation; it vanishes when planned replacement costs 50% more (`C_batt_planned x1.5`: -13) and when emergency replacement is 50% cheaper (0). **The economics hinge on the gap between planned and emergency replacement cost**, which the placeholders only crudely assume. This is the first number to get from an operator.
- Base rate: saving 51 at x0.25, 165 at x0.5, 354 at x1, 491 at x1.5. Break-even (`breakeven_bboxx.png`): at a tenth of the as-sampled base rate (about 0.3% per month) the policy still pays if a trip costs at most about the placeholder value, and does not pay at 0.05x or at 2x trip cost.

## Limitations
Hypothetical costs; as-sampled prevalence; younger test cohort; 56-day true-alert window and 90-day cooldown are assumptions; confirmation sensitivity (0.90) and false confirm (0.10) are assumptions; a true alert that is not confirmed or dispatched counts as missed without a second chance; labels are customer repairs, so some "false" dispatches may be genuine incipient faults; one GP hyperparameter set per fold; no leave-district-out.
