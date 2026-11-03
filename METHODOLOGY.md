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
See Decisions entry of 2026-11-03 and `docs/phase4_design.md`.

## Assumptions and limitations
Lead-acid, not LFP; one manufacturer; alive batteries are censored; labels are customer repair decisions, not confirmed battery faults; hyperparameters poorly determined by 8 batteries; CC BY-NC data (research use only).

## Decisions
- **2026-11-02 (Phase 2)** Per-segment offset `tau` added; chosen over letting the time-process noise grow to its bound (log-posterior +1,700). Dropped sigma_f half-normal prior (units not comparable to Aitio).
- **2026-11-02 (Phase 2)** 5-min grid instead of 1-min for speed; Aitio Table 2 reports no loss.
- **2026-11-02 (Phase 3)** Truncate failed batteries at the repair date. Per-fold hyperparameters to avoid test leakage. `unc` and `stale_days` excluded (data-availability leak; `stale_days` alone gives 69% at h = 0).
- **2026-11-03 (Phase 4)** Documents `CLAUDE.md`, `LOG.md`, `METHODOLOGY.md` reconstructed; repo published to GitHub (private) so it is accessible and sharable. Code-only: data stays out of git (CC BY-NC).
