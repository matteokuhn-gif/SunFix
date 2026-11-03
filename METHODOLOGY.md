# Methodology (scientific record)

Reconstructed on 2026-11-03 from the git history, `docs/` reports and code, because it was not created in Phase 0 (see LOG.md, session 5). Details live in `docs/aitio_protocol.md`, `docs/battgp_notes.md`, `docs/phase2_report.md`, `docs/phase3_report.md`. Update this file whenever a method changes; never edit retroactively without noting it.

## Research questions
1. Reproduction: how well does BattGP's GP resistance model predict battery end of life (EOL) on the Bboxx data at failure, 4 and 8 weeks before? Benchmark (Aitio & Howey 2021): 82% / 73% at 0 / 56 days.
2. Minimum data: how much does accuracy degrade as telemetry is thinned (sampling interval, temperature, history, precision, packet loss)? What is the smallest specification that gives a useful 30-day warning (balanced accuracy >= 70%)?

## Data and cleaning rules (`sunfix/load.py`, `sunfix/clean.py`)
- Negative current = charging. 1,027 batteries; 491 failed (entered repair), 536 alive (censored).
- Keep activation <= t <= repair date (failed) or end of data (alive). Telemetry continues a median 89 days after repair for failed batteries and does not belong to the failed battery.
- Drop NaN, voltage outside 9-16 V (physically impossible for a 12 V VRLA), duplicate timestamps; sort by time.
- Charging segments per Aitio Table 1: duration > 6000 s, no gap > 610 s, rest voltage before the segment 11.5-12.9 V with |I| < 0.1 A, max V > 14 V. Truncated at first V > 14 V, PCHIP-interpolated to a 1-min grid. Rows with |I| < 0.2 A dropped for the resistance model.

## Model (`sunfix/ocv.py`, `soc.py`, `resistance.py`, `fit_hyper.py`)
- BattGP `ApproxSpatioTemporalGP` (Wiener-velocity time kernel + SE over operating point on 20 k-means basis points + reference point), one "cell" per battery, causal filter only. Observation R = (V - OCV)/|I| on a 5-min grid, heteroskedastic noise.
- BattGP is not OCV-free in practice here. Two variants: `ocv` (Coulomb-counted SOC + GITT OCV curve, operating point I, SOC, T; main) and `nocv` (rest voltage before segment as OCV, operating point I, T).
- Extension: per-segment random offset `tau`. No sigma_f half-normal prior. Hyperparameters fitted label-free on 8 training-fold batteries (4 failed, 4 alive) per fold, Nelder-Mead. Gassing constants are generic VRLA assumptions.

## Labels and evaluation protocol (`sunfix/classify.py`)
- Label: battery failed (customer repair decision). Predict "fails within horizon h" via decision time = series end minus h days, h in {0, 14, 28, 42, 56}. Series end = repair date (failed) or last telemetry (alive).
- 5-fold stratified CV by battery (seed 0); hyperparameters fitted inside each training fold; pooled out-of-fold predictions; threshold 0.5.
- Deviations from Aitio: Aitio's extra test subsets (40/60/80% of failures) not reproduced; random-walk benchmark model not implemented.

## Features (`sunfix/features.py`)
- Resistance (`res_core`, headline): level, slope now, slopes over 2/4/8 weeks, level relative to own days 60-120 baseline. Excluded as leaking: `unc`, `stale_days`.
- Usage/stress: age, Ah throughput, mean T, mean V, cycles, float hours.

## Metrics
Balanced accuracy (headline), sensitivity, specificity, precision, false alarms per 1,000 per month, Brier, ECE, 95% bootstrap CI, fold-to-fold sd. Usage-only baseline (about 70.5%) is reported alongside every result.

## Minimum-data experiment design (Phase 4)
Single-factor degradations of the cleaned data, scenario list and implementation in `sunfix/phase4.py`; details, limitations and results in `docs/phase4_report.md`. Decision times fixed from native data; Phase 3 per-fold hyperparameters reused (refit only without temperature); h = 28 and 56.

## Phase 6 (false-positive economics, benchmarks, LFP transfer)
### 6a Cost-aware evaluation
Landmarks over all system-days (age >= 120 d), label "fails within 30 d", alive batteries censored; temporal split by activation date (50/20/30% train/calibration/test); gradient boosting + Platt calibration; alert policy (threshold, persistence N, confirmation, capacity K, 90-day cooldown); expected-value cost accounting with hypothetical costs in `config/costs.yaml`; operating point selected on the calibration cohort, reported on test with a battery-level bootstrap; baselines reactive, age-based, Elkan. Details and limitations: `docs/phase6a_report.md`.

### 6b Benchmark models
Extended no-look-ahead features (`sunfix/features6.py`), models and evaluation in `sunfix/bench.py` and `docs/phase6b_report.md`. Headline metric is balanced accuracy restricted to age >= 360 d (the at-risk period), because over all landmarks the metric is dominated by trivially healthy young systems.

### 6c Conformal gate and persistence
Split-conformal thresholds on healthy at-risk system-days of the calibration cohort (static, age-conditional, adaptive with 56-day label delay, sliding window); realised false-alarm rate by month on the test cohort; persistence evaluated as the minimum score over N days against i.i.d. theory. Details: `docs/phase6c_report.md`.

## Assumptions and limitations
Lead-acid, not LFP; one manufacturer; alive batteries are censored; labels are customer repair decisions, not confirmed battery faults; hyperparameters poorly determined by 8 batteries; CC BY-NC data (research use only).

## Decisions
- **2026-11-02 (Phase 2)** Per-segment offset `tau` added; chosen over letting the time-process noise grow to its bound (log-posterior +1,700). Dropped sigma_f half-normal prior (units not comparable to Aitio).
- **2026-11-02 (Phase 2)** 5-min grid instead of 1-min for speed; Aitio Table 2 reports no loss.
- **2026-11-02 (Phase 3)** Truncate failed batteries at the repair date. Per-fold hyperparameters to avoid test leakage. `unc` and `stale_days` excluded (data-availability leak; `stale_days` alone gives 69% at h = 0).
- **2026-11-03 (Phase 4)** Documents `CLAUDE.md`, `LOG.md`, `METHODOLOGY.md` reconstructed; repo published to GitHub (private) so it is accessible and sharable. Code-only: data stays out of git (CC BY-NC).
- **2026-11-03 (Phase 4)** Ran all scenarios on a fixed stratified 40% subsample of batteries (seed 0) with the Phase 3 folds restricted to it, instead of all 1,027, to cut runtime from about 70 to about 25 minutes. Alternative: full run (rejected for time). Cost: intervals of about +-4 points. Full-data runs exist only for 5-min sampling and daily summaries.
- **2026-11-03 (Phase 4)** Daily-summary scenario uses a proxy (Vmax-Vmin)/(Imax-Imin) because the resistance GP needs charging curves. Segment gap limit scaled to 2.5x the sampling interval for thinned data. Short-history baseline for relative level = first quarter of the visible window.
- **2026-11-03 (Phase 4)** Empty history windows (telemetry gaps) count as "no reading" (median-imputed; coverage column reports the share with a reading).
- **2026-11-03 (6a)** Temporal split by activation date (50/20/30) instead of random folds, as required by the brief; leave-one-district-out skipped because no location metadata exist. Alternative (random folds) rejected: leaks calendar-time effects.
- **2026-11-03 (6a)** Alive batteries' last 30 d (labels) and last 56 d (simulation exposure) dropped as right-censored. Alternative: treat as healthy (rejected: biases hazard down).
- **2026-11-03 (6a)** Expected-value simulation for confirmation and capacity (no random draws); un-dispatched true alerts count as missed. Operating point chosen on calibration cohort, never on test. Platt chosen over isotonic as the headline calibrator (fit on 205 batteries; isotonic reported).
- **2026-11-03 (6a)** Base-rate sensitivity implemented by re-weighting failed batteries because the sample is about 48% failed.
- **2026-11-03 (6b)** Fleet z-score uses only peers installed up to 60 days BEFORE the battery (causal) instead of the brief's +-1 month window, which would use peers' later data. Bayesian online change-point detection not implemented (CUSUM only), to be added only if the unsupervised route proves useful (it did not: 58% at-risk).
- **2026-11-03 (6b)** Headline benchmark metric restricted to age >= 360 d, threshold re-chosen on the calibration cohort. Alternative (all landmarks) rejected: inflated by age (88-91%).
- **2026-11-03 (6b)** Cox and RSF trained on landmarks every 14 days (about 18k rows) with alive batteries censored; GPC on 2,000 sampled landmarks; availability features kept out of headline models (ablation only).
- **2026-11-03 (env)** numpy stays pinned at 1.26.4 (BattGP/torch code breaks on numpy 2.5); scikit-survival 0.28 needs `np.trapezoid`, shimmed in `sunfix/bench.py`.
- **2026-11-03 (6c)** Persistence implemented as a conformal statistic (minimum score over the last N days) so each N has its own calibrated threshold and false-alarm budgets are equal across N. Healthy = no failure within 56 days with the outcome observed, age >= 360 d. Online variants use healthy outcomes only from dates <= t - 56 (labels arrive late).
