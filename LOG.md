# Lab notebook (append-only)

Sessions 1-4 were not logged at the time. Entries below were reconstructed on 2026-11-03 from git history and are marked (reconstructed).

## 2026-11-02 19:23 – Session 1 (Phase 0) (reconstructed)
**Goal:** setup, BattGP and Aitio notes. **Done:** repo, venv, `docs/battgp_notes.md`, `docs/aitio_protocol.md`. **Commit:** 93cf3f8, 9b379b0

## 2026-11-02 19:33 – Session 2 (Phase 1) (reconstructed)
**Goal:** data pipeline. **Done:** loader, cleaning, Aitio segment extraction, EDA (`docs/eda_report.md`). **Commit:** bbea8e2, 9e2b6d8

## 2026-11-02 19:46 – Session 3 (Phase 2) (reconstructed)
**Goal:** BattGP resistance model on 60 dev batteries. **Results:** `docs/phase2_report.md`; ocv variant predicts voltage 26% better than nocv; 0.76 s/battery. **Decisions:** per-segment offset. **Commit:** aad7bee

## 2026-11-02 20:13 – Session 4 (Phase 3) (reconstructed)
**Goal:** EOL prediction on all 1,027 batteries. **Results:** balanced accuracy 81 / 79 / 79 / 76 / 74% at h = 0/14/28/42/56 (paper 82/80/76/76/73); usage-only about 70.5%. **Problems:** `stale_days` leak found and removed. **Commit:** 531d19e

## 2026-11-03 – Session 5 (Phase 4 start)
**Goal:** reconcile documents (CLAUDE.md, LOG.md, METHODOLOGY.md were never created), publish repo to GitHub, design and start Phase 4.
**Done:** created the three documents; see below for continuation.
**Problems / surprises:** the three files were missing although expected; reconstructed. The original brief was missing from the folder until the user added it.
**Done:** Phase 4 harness (`sunfix/phase4.py`, `phase4_report.py`), 15 scenarios on a 40% stratified subsample; report `docs/phase4_report.md`; repo pushed to GitHub (private, matteokuhn-gif/SunFix).
**Results:** no scenario measurably below native (full model 77-81% at 28 d vs 78% native, +-4 pt); usage-only 61-72%; one daily record gives 79% (subsample) / 75% (full data, 56 d: 70.8%). `results/minimum_data_table_sub40.csv`, `results/figures/phase4_minimum_data_sub40.png`.
**Problems / surprises:** joblib workers did not see runtime-registered feature sets (moved to `features.SETS`); empty 30-day windows crashed stress features (guard added). Run crashed once and was restarted; finished scenarios were kept.
**Decisions:** subsample, daily proxy, gap scaling (see METHODOLOGY.md).
**Next:** Phase 5 REPORT.md; optionally a combined cheap-device scenario and a full-data run of the key scenarios; Phase 6.
**Commit:** see git log

## 2026-11-03 – Session 6 (Phase 5, Phase 6a)
**Goal:** write REPORT.md; cost-aware evaluation harness (6a); stop after 6a per PHASE6_PROMPT.md.
**Done:** REPORT.md; `config/costs.yaml` (hypothetical); `sunfix/landmarks.py`, `eval_ops.py`, `run_6a.py`; `docs/phase6a_report.md`; figures cost_vs_threshold, leadtime_hist, calibration_bboxx, breakeven_bboxx.
**Results:** 434k landmarks; hazard 0 below 360 d, 4.9-14% per month above (as sampled). Phase-3-style operating point: 67 false vs 17 true dispatches per 1,000 per month (precision 21%), cost above reactive under hypothetical costs. Best cost-optimal policy about 9% below reactive (hypothetical). Calibration test ECE 0.006.
**Problems / surprises:** age-based baseline equals reactive (placeholder planned/emergency costs too close; test cohort younger). Sample is 48% failed, so as-sampled hazard overstates a real fleet. No district metadata: leave-one-district-out skipped.
**Decisions:** see METHODOLOGY.md Phase 6 and Decisions.
**Next:** 6b benchmark models (features.py with no look-ahead, LightGBM hazard, Cox/RSF, CUSUM, ensemble).
**Commit:** see git log
