# Phase 6c report: conformal gate and persistence on autocorrelated scores

Run: `python -m sunfix.conformal` (under 1 min). Outputs: `results/conformal_far_by_month.csv`, `results/persistence_bboxx.csv`, `figures/conformal_far_by_month.png`, `figures/persistence_effect.png`. No cost values are used; nothing here depends on `config/costs.yaml`.

## Set-up
Score = 30-day LightGBM probability (6b model, trained on the training cohort only). "Healthy system-day" = no failure within 56 days with the outcome observed, age >= 360 d. Calibration = healthy days of the calibration cohort (21.7k days, Mar-Aug 2020); evaluation = healthy days of the later-activated test cohort (15.3k days, Apr-Sep 2020; Apr and Sep are tiny). Split-conformal threshold = the ceil((n+1)(1-alpha))/n quantile of healthy calibration scores. Variants: static; age-conditional (Mondrian, 60-day age bins); adaptive conformal inference with delayed labels (outcomes only known 56 days later, gamma 0.02); sliding window (previous 90 days of healthy outcomes, 56-day delay). The days of one battery are strongly autocorrelated, so the effective sample size is far smaller than 15k.

## Conformal gate: realised false-alarm rate on later months
Overall realised false-alarm rate on healthy system-days (target alpha in the first column):
| alpha | static | age-conditional | adaptive (ACI) | sliding window |
|---|---|---|---|---|
| 0.01 | 1.0% | 0.9% | 1.4% | 5.4% |
| 0.02 | 1.5% | 1.6% | 2.3% | 7.5% |
| 0.05 | 3.7% | 4.7% | 6.2% | 14.2% |
| 0.10 | 8.7% | 10.1% | 13.0% | 24.4% |
- The static gate is conservative overall (at or below target) because the test cohort is younger and healthier than the calibration cohort, but **it drifts upward with time**: at alpha = 0.05 the monthly rate is 2.5% (May), 1.8% (Jun), 5.0% (Jul) and 6.2% (Aug): about 2-3 times higher in July-August than in May-June, with a dip in between. The likely driver is battery ageing (older systems score higher), not season: the test window is only five months and cohort age and calendar month move together, so **seasonality cannot be separated from age** with this data. A year or more of fleet data with a stable age mix is needed.
- Age-conditional calibration tracks the target best (4.7% / 10.1% for 5% / 10%).
- Adaptive conformal with 56-day label delay overshoots slightly (6.2%, 13.0%): feedback arrives too late to follow a four-month drift.
- The sliding window is clearly worse (14% at a 5% target; May 41%): its pool is dominated by younger, lower-scoring batteries from earlier months, so thresholds are too low once older systems enter. Not recommended in this form.
- Exchangeability (the conformal guarantee) does not hold across ages and months; the empirical results above are what to expect, not the guarantee.

## Persistence on autocorrelated scores (equal-error threshold, as in the i.i.d. comparison)
At the N=1 equal-error threshold the healthy false-positive rate is 19.6%.
| N | Empirical FPR | i.i.d. theory (p^N) | Empirical / i.i.d. | Cited (Adnan et al. 2011) |
|---|---|---|---|---|
| 1 | 19.6% | 19.6% | 1 | 25.3% |
| 3 | 17.2% | 0.7% | 23x | 6.4% |
| 5 | 15.4% | 0.03% | 540x | 1.0% |
| 7 | 13.9% | n/a | n/a | n/a |
Persistence cuts the false-positive rate by only 12%, 21% and 29% at N = 3, 5, 7, against 75% and 96% reductions in the cited i.i.d. figures (N = 3, 5). Reason: a healthy battery with a high score tends to keep it for days, so consecutive days are not independent evidence. (I use the cited numbers only as quoted in the project brief; I have not checked the definition of their scheme, and p^N does not reproduce their numbers either.)

## Persistence at an equal false-alarm budget (conformal threshold per N; 93 failed test batteries)
At alpha = 0.05: detection of failed batteries 65.6% (N=1), 67.7%, 68.8%, 67.7% (N=3, 5, 7); median warning 30, 29, 28, 26 days; share of warnings at least 30 days ahead 51%, 49%, 48%, 41%; healthy batteries ever alerted 11.1%, 10.4%, 10.8%, 10.8%. At alpha = 0.10 the picture is the same (detection about 83-84%, median lead 40.5 down to 37 days). **Persistence did not improve detection or reduce battery-level false alarms, and shortened the warning by a few days.** The detection differences are within noise (93 failures).

## What it means
1. Use a conformal threshold to set the false-alarm budget explicitly, preferably calibrated by age band, and re-check it monthly: even over five months the realised rate moved by a factor of 2-3.
2. Do not rely on "N days in a row" to remove false alarms: with these scores it removes little, because the errors are persistent, not random. If false alarms must come down, raise the threshold or add an independent check (remote confirmation, a second signal) rather than waiting longer.
3. The 6a assumption that persistence lowers false dispatches was optimistic; 6a's grid already showed little gain from N beyond a small effect at low thresholds.

## Limitations
Five months of at-risk test data; seasonality confounded with age; effective sample size small and autocorrelated; one score model; labels arrive 56 days late in the online variants; no leave-district-out. Cited i.i.d. numbers unverified.
